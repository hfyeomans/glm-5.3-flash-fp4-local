# Native TensorFold 1.1.3 + pinned PR #1050, targeting RTX PRO 6000 Blackwell (SM120).
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:69e9e39eb8fe2cda271654a0f5eac2f1bb946b2fb9c460eb19c7c3c155f4e64e AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils build-essential libnccl2=2.31.2-1+cuda13.3 && rm -rf /var/lib/apt/lists/*
ARG ZIG_SHA256=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026
RUN curl -fL https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz && echo "$ZIG_SHA256  /tmp/zig.tar.xz" | sha256sum -c - && mkdir /opt/zig && tar -xJf /tmp/zig.tar.xz --strip-components=1 -C /opt/zig && rm /tmp/zig.tar.xz
ENV PATH="/opt/zig:${PATH}"
ARG TF_COMMIT=36415df4f76e4458e82d0fa729ecb28e287ebf38
ARG PR1050_COMMIT=9b23b6a2e7ff226ca61934189c4cd57bf2933980
ARG TF_MERGED_TREE=1d52effd182d9319d775d5787904c36783e532f1
ARG XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa
RUN git clone --filter=blob:none --depth 256 --branch v1.1.3 https://github.com/ashhart/TensorFold.git /src/tensorfold && test "$(git -C /src/tensorfold rev-parse HEAD)" = "$TF_COMMIT"
RUN git -C /src/tensorfold fetch --depth 128 https://github.com/Bizuayeu/TensorFold.git "$PR1050_COMMIT" && test "$(git -C /src/tensorfold rev-parse FETCH_HEAD)" = "$PR1050_COMMIT" && test "$(git -C /src/tensorfold merge-tree --write-tree "$TF_COMMIT" "$PR1050_COMMIT")" = "$TF_MERGED_TREE" && git -C /src/tensorfold read-tree --reset -u "$TF_MERGED_TREE"
RUN git clone --depth 1 --branch v0.2.8 --recurse-submodules --shallow-submodules https://github.com/mlc-ai/xgrammar.git /src/xgrammar && test "$(git -C /src/xgrammar rev-parse HEAD)" = "$XGRAMMAR_COMMIT"
WORKDIR /src/tensorfold
ARG BUILD_JOBS=4
RUN test "$(zig version)" = "$(cat .zig-version)" && zig build native grammar compile-cuda-tests -Dversion=1.1.3 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=120 -Dxgrammar=/src/xgrammar -j${BUILD_JOBS}
RUN zig build test-grammar-install-host -j${BUILD_JOBS} && zig test zig/src/core/nvfp4_csf32.zig && zig test zig/src/core/modelopt.zig
RUN mkdir -p /opt/tensorfold/bin && find .zig-cache/o -type f -name tf-cuda-test -executable -exec cp {} /opt/tensorfold/bin/tf-cuda-test \; && test -x /opt/tensorfold/bin/tf-cuda-test && cp -a zig-out/native/. /opt/tensorfold/ && cp -a LICENSE* THIRD_PARTY_NOTICES.md /opt/tensorfold/ && mkdir /opt/tensorfold/licenses && cp /src/xgrammar/LICENSE /opt/tensorfold/licenses/xgrammar-LICENSE && cp /src/xgrammar/NOTICE /opt/tensorfold/licenses/xgrammar-NOTICE && cp /src/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/licenses/dlpack-LICENSE && cp -a /src/xgrammar/3rdparty/picojson /opt/tensorfold/licenses/picojson
RUN /opt/tensorfold/bin/tensorfold-native --version && /opt/tensorfold/bin/tensorfold-native capabilities --json > /opt/tensorfold/capabilities.json && dpkg-query -W > /opt/tensorfold/build-packages.txt
# Retain compiler and sources for repeatable host diagnostics in this first build.
FROM build AS runtime
LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold"       org.opencontainers.image.revision="36415df4f76e4458e82d0fa729ecb28e287ebf38"       org.opencontainers.image.version="1.1.3-pr1050"       lab.glm53.tensorfold.pr1050="9b23b6a2e7ff226ca61934189c4cd57bf2933980"       lab.glm53.tensorfold.tree="1d52effd182d9319d775d5787904c36783e532f1"       lab.glm53.tensorfold.recipe="native-fp4-tp2"
COPY scripts/loader_probe.zig /opt/recipe/loader_probe.zig
RUN zig build-exe --dep core -Mroot=/opt/recipe/loader_probe.zig --dep sys --dep checkpoint_config --dep save_guard -Mcore=/src/tensorfold/zig/src/core/root.zig --dep checkpoint_config --dep save_guard -Msys=/src/tensorfold/zig/src/platform/sys.zig --dep save_guard -Mcheckpoint_config=/src/tensorfold/zig/src/core/checkpoint_config.zig -Msave_guard=/src/tensorfold/zig/src/core/save_guard.zig -lc -femit-bin=/opt/tensorfold/bin/glm-loader-probe
COPY scripts/runner.sh /opt/recipe/runner.sh
ENV XDG_CACHE_HOME=/cache NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt/tensorfold
ENTRYPOINT ["/opt/recipe/runner.sh"]
