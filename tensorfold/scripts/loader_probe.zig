//! Exercise the exact core NVFP4 reader called by GLM's CUDA loader, without a GPU.
const std = @import("std");
const core = @import("core");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedModelDirectory;
    var ck = try core.Checkpoint.openModel(init.gpa, init.io, args[1]);
    defer ck.close();
    const name = "model.language_model.layers.3.mlp.experts.0.gate_proj";
    const linear = core.nvfp4.Linear.read(&ck, name) catch |err| {
        std.debug.print("FAIL native GLM NVFP4 reader: {s}: {s}\n", .{ name, @errorName(err) });
        return 1;
    };
    std.debug.print("PASS native GLM NVFP4 reader: {s}: {d} x {d}\n", .{ name, linear.n, linear.k });
    return 0;
}
