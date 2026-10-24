//! Check mixed recipes and a real CSF32 scale matrix without a GPU; GPU loader gate is separate.
const std = @import("std");
const core = @import("core");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.ExpectedModelDirectory;
    const path = try std.fs.path.join(a, &.{ args[1], "config.json" });
    const config = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 24));
    var why: core.modelopt.Why = .{};
    var recipes = try core.modelopt.admitRecipes(init.gpa, config, &why);
    defer recipes.deinit();
    var ck = try core.Checkpoint.openModel(init.gpa, init.io, args[1]);
    defer ck.close();
    const module = "model.language_model.layers.3.mlp.experts.0.gate_proj";
    if (recipes.of(module) != .nvfp4_csf) return error.WrongRecipe;
    const w = try ck.get(module ++ ".weight");
    if (w.dtype != .u8 or w.rank != 2) return error.WrongWeight;
    var parts: core.nvfp4_csf32.Parts = undefined;
    inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }) |part| {
        const tensor = try ck.get(module ++ ".weight_scale.csf32_" ++ part);
        @field(parts, part) = tensor.bytes;
    }
    const columns = w.shape[1] / 8;
    const scales = try a.alloc(u8, w.shape[0] * columns);
    try core.nvfp4_csf32.decode(parts, w.shape[0], columns, scales);
    var encoded = try core.nvfp4_csf32.encode(init.gpa, scales, w.shape[0], columns);
    defer encoded.deinit(init.gpa);
    const back = encoded.parts();
    inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }) |part| {
        if (!std.mem.eql(u8, @field(parts, part), @field(back, part))) return error.RoundTripMismatch;
    }
    std.debug.print("PASS mixed recipes: {d} modules; real CSF32 decode/re-encode: {d} scales, five parts byte-identical. GPU loader gate still required.\n", .{ recipes.modules.count(), scales.len });
    return 0;
}
