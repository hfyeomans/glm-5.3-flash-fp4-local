// Adapted from TensorFold glm_mm_bench.zig at the pinned PR tree; Apache-2.0.
//! GLM's bf16 linears on one GPU: the family's b16.matmul against glm_mm (tiled weights) on a rank's shapes, by rows.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// A rank's linears of GLM-5.3-Flash at two ranks (n, k): KDA projections and output, MLA, shared expert, dense layer.
const shapes = [_][2]usize{ .{ 6416, 4096 }, .{ 4096, 2048 }, .{ 2048, 4096 }, .{ 4096, 1024 }, .{ 12288, 4096 }, .{ 4096, 6144 }, .{ 2560, 4096 }, .{ 4096, 4096 } };
const reps = 30;

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    var mm_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer mm_mod.unload();
    const glue = try glm.glue.Kernels.load(gm);
    const set: glm.glue.Set = .{ .glue = &glue };
    const mk = try glm.mm.Kernels.load(mm_mod);
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    var total = [2][7]f64{ @splat(0), @splat(0) };
    for (shapes) |sh| {
        const n = sh[0];
        const k = sh[1];
        const host_w = try a.alloc(u16, n * k);
        defer a.free(host_w);
        for (host_w) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) * 2 - 1) * 0.02)) >> 16);
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_w));
        defer w.free();
        var wt = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
        defer wt.free();
        try glm.mm.tile(&mk, s, w.ptr, wt.ptr, n, k);
        const tw: glm.mm.Tiled = .{ .w = wt.ptr, .n = n, .k = k };
        const host_x = try a.alloc(u16, 512 * k);
        defer a.free(host_x);
        for (host_x) |*v| v.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
        const reference = try a.alloc(f64, 3 * n);
        defer a.free(reference);
        const tolerance = try a.alloc(f64, 3 * n);
        defer a.free(tolerance);
        for (0..3) |row| for (0..n) |col| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..k) |i| {
                const xv: f64 = @as(f32, @bitCast(@as(u32, host_x[row * k + i]) << 16));
                const wv: f64 = @as(f32, @bitCast(@as(u32, host_w[col * k + i]) << 16));
                sum += xv * wv;
                magnitude += @abs(xv * wv);
            }
            reference[row * n + col] = sum;
            tolerance[row * n + col] = magnitude * @as(f64, @floatFromInt(k + 8)) * 0x1p-24 + 1e-30;
        };
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_x));
        defer x.free();
        var out_a = try cuda.DeviceBuffer.alloc(gpu.d, 512 * n * 4);
        defer out_a.free();
        var out_b = try cuda.DeviceBuffer.alloc(gpu.d, 512 * n * 4);
        defer out_b.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, @max(glm.b16.partBytes(127, n, k), glm.mm.partBytes(128, tw))));
        defer part.free();
        var ms: [2][7]f64 = undefined;
        for ([_]usize{ 1, 2, 4, 8, 64, 128, 512 }, 0..) |rows, ri| {
            for (0..2) |which| {
                for (0..2) |round| {
                    try t0.record(s);
                    for (0..reps) |_| {
                        if (which == 0) try glm.b16.matmul(set, s, x.ptr, k, w.ptr, out_a.ptr, part.ptr, rows, n, k, true) else try glm.mm.matmul(&mk, s, x.ptr, k, tw, out_b.ptr, part.ptr, rows, true, false);
                    }
                    try t1.record(s);
                    try t1.synchronize();
                    if (round == 1) ms[which][ri] = @as(f64, try cuda.Event.elapsedMs(t0, t1)) / reps;
                }
                total[which][ri] += ms[which][ri];
            }
            // the two products agree (different summation orders: a few fp32 ulps of the row's magnitude)
            const ga = try a.alloc(f32, rows * n);
            defer a.free(ga);
            const gb = try a.alloc(f32, rows * n);
            defer a.free(gb);
            try out_a.download(0, std.mem.sliceAsBytes(ga));
            try out_b.download(0, std.mem.sliceAsBytes(gb));
            var worst: f64 = 0;
            var worst_col: usize = 0;
            var max_abs: f64 = 0;
            for (0..@min(rows, 3)) |row| for (0..n) |col| {
                const index = row * n + col;
                const difference = @abs(@as(f64, gb[index]) - reference[index]);
                const fraction = difference / tolerance[index];
                max_abs = @max(max_abs, difference);
                if (fraction > worst) { worst = fraction; worst_col = col; }
                try check.expect(std.math.isFinite(fraction), "glm-mm-fp64: finite error", .{});
            };
            std.debug.print("FP64 target={d} n={d} k={d} rows={d} checked_rows={d} worst_col={d} fraction={e:.9} max_abs={e:.9}\n",
                .{ cuda.sm120_tuning.splitTarget(), n, k, rows, @min(rows, 3), worst_col, worst, max_abs });
            try check.expect(worst <= 1, "glm-mm-fp64: independent reference bound", .{});
            var most: f32 = 0;
            var big: f32 = 0;
            for (ga, gb) |u, v| {
                most = @max(most, @abs(u - v));
                big = @max(big, @abs(u));
            }
            try check.expect(most <= big * 1e-3, "glm-mm-bench: {d} x {d} at {d} rows: the products agree", .{ n, k, rows });
        }
        std.debug.print("RESULT {d} x {d} (rows 1/2/4/8/64/128/512):\n  b16    {d:.3}/{d:.3}/{d:.3}/{d:.3}/{d:.3}/{d:.3}/{d:.3} ms\n  glm_mm {d:.3}/{d:.3}/{d:.3}/{d:.3}/{d:.3}/{d:.3}/{d:.3} ms\n", .{ n, k, ms[0][0], ms[0][1], ms[0][2], ms[0][3], ms[0][4], ms[0][5], ms[0][6], ms[1][0], ms[1][1], ms[1][2], ms[1][3], ms[1][4], ms[1][5], ms[1][6] });
    }
    std.debug.print("RESULT all shapes once (rows 1/2/4/8/64/128/512):\n  b16    {d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2} ms\n  glm_mm {d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2} ms\n", .{ total[0][0], total[0][1], total[0][2], total[0][3], total[0][4], total[0][5], total[0][6], total[1][0], total[1][1], total[1][2], total[1][3], total[1][4], total[1][5], total[1][6] });
    check.pass("glm-mm-bench: both products agree on every shape and row count", .{});
}
