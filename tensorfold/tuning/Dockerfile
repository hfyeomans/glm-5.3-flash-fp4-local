# Local experiments only; the qualified image remains a separate immutable baseline.
FROM glm53-fp4-tensorfold:1.1.3-pr1050-sm120-context
ARG PATCH_SHA256=4e1cf2a5666d6ddb1527bbf7192f785caa928614feb68ec2912608a95d8cfdef
ARG TUNED_TREE=0c2bc6ddf016c0d96d63fbd7d1238b6355603578
WORKDIR /src/tensorfold
COPY sm120-experiments.patch /opt/recipe/sm120-experiments.patch
RUN echo "$PATCH_SHA256  /opt/recipe/sm120-experiments.patch" | sha256sum -c - && test "$(git write-tree)" = 1d52effd182d9319d775d5787904c36783e532f1 && git apply --check /opt/recipe/sm120-experiments.patch && git apply --index /opt/recipe/sm120-experiments.patch && test "$(git write-tree)" = "$TUNED_TREE"
RUN zig test zig/src/cuda/sm120_tuning.zig -lc
RUN touch /tmp/tuning-build-start && zig build native compile-cuda-tests -Dversion=1.1.3 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dxgrammar=/src/xgrammar -j4 && find .zig-cache/o -type f -name tf-cuda-test -newer /tmp/tuning-build-start > /tmp/test-binaries && test "$(wc -l < /tmp/test-binaries)" = 1 && cp "$(cat /tmp/test-binaries)" /opt/tensorfold/bin/tf-cuda-test && cp -a zig-out/native/. /opt/tensorfold/ && sha256sum /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/bin/tf-cuda-test > /opt/tensorfold/tuning-binaries.sha256
LABEL org.opencontainers.image.version="1.1.3-pr1050-local-experiments" lab.glm53.tensorfold.base_tree="1d52effd182d9319d775d5787904c36783e532f1"
LABEL lab.glm53.tensorfold.tree="$TUNED_TREE" lab.glm53.tensorfold.tuning="$PATCH_SHA256"
WORKDIR /opt/tensorfold
