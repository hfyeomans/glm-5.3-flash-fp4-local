# Experimental CSF32 residency; the qualified normal TensorFold and vLLM images stay unchanged.
FROM glm53-fp4-tensorfold:sm120-tuning-v1
ARG PATCH_SHA256=567b5953034033ed5432fc31255149c9315d5e7fc93e1aaaffbc2f05c5d46905
ARG RESIDENT_TREE=b02cfca0f02235fa8f4b87c0ae921664a73d1807
WORKDIR /src/tensorfold
COPY resident-scales.patch /opt/recipe/resident-scales.patch
RUN echo "$PATCH_SHA256  /opt/recipe/resident-scales.patch" | sha256sum -c - && test "$(git write-tree)" = 0c2bc6ddf016c0d96d63fbd7d1238b6355603578 && git update-index --refresh && git apply --check /opt/recipe/resident-scales.patch && git apply --index /opt/recipe/resident-scales.patch && test "$(git write-tree)" = "$RESIDENT_TREE"
RUN zig test zig/src/core/nvfp4_csf32_slice.zig && zig test zig/src/cuda/nvfp4_separate.zig -lc && zig test zig/src/cuda/nvfp4_csf32.zig -lc
RUN touch /tmp/csf-build-start && zig build native compile-cuda-tests -Dversion=1.1.3 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dxgrammar=/src/xgrammar -j4 && find .zig-cache/o -type f -name tf-cuda-test -newer /tmp/csf-build-start > /tmp/csf-test-binaries && test "$(wc -l < /tmp/csf-test-binaries)" = 1 && cp "$(cat /tmp/csf-test-binaries)" /opt/tensorfold/bin/tf-cuda-test && cp -a zig-out/native/. /opt/tensorfold/ && sha256sum /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/bin/tf-cuda-test > /opt/tensorfold/csf-binaries.sha256
LABEL org.opencontainers.image.version="1.1.3-pr1050-csf-resident-experiment-v1" lab.glm53.tensorfold.base_tree="1d52effd182d9319d775d5787904c36783e532f1"
LABEL lab.glm53.tensorfold.tree="$RESIDENT_TREE" lab.glm53.tensorfold.tuning="$PATCH_SHA256" lab.glm53.tensorfold.scale_layout="2"
WORKDIR /opt/tensorfold
