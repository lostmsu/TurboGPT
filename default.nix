{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; }
, arch ? 86
, threads ? 512
, tile ? 32
, pipeline ? false
}:

let
  cuda = pkgs.cudaPackages_12_9;
  gitDir = ./.git;
  src = pkgs.lib.cleanSourceWith {
    src = ./.;
    name = "turbogpt-source";
    filter = path: type:
      let base = baseNameOf (toString path);
      in
      if type == "directory"
      then !builtins.elem base [ ".git" "build" "__pycache__" ]
      else type != "symlink" && !pkgs.lib.hasSuffix ".pyc" (toString path);
  };
  commonFlags = [
    "-O3"
    "-std=c++17"
    "--use_fast_math"
    "-lineinfo"
    "--threads"
    "0"
    "-gencode"
    "arch=compute_${toString arch},code=sm_${toString arch}"
    "-DTURBOGPT_THREADS=${toString threads}"
    "-DTURBOGPT_TILE=${toString tile}"
    "-DTURBOGPT_PIPELINE=${if pipeline then "1" else "0"}"
  ];
  escapedFlags = map pkgs.lib.escapeShellArg commonFlags;
in
pkgs.stdenv.mkDerivation {
  pname = "turbogpt";
  version = "unstable";

  inherit src;

  nativeBuildInputs = [
    cuda.cuda_nvcc
    cuda.cuda_cudart
    cuda.cuda_cccl
    pkgs.git
  ];

  buildPhase = ''
    runHook preBuild

    revisionTmp=$(mktemp -d)
    if [ -f ${gitDir}/index ]; then
      cp ${gitDir}/index "$revisionTmp/index"
    fi
    mkdir -p "$revisionTmp/objects"
    export GIT_INDEX_FILE="$revisionTmp/index"
    export GIT_OBJECT_DIRECTORY="$revisionTmp/objects"
    export GIT_ALTERNATE_OBJECT_DIRECTORIES="${gitDir}/objects"
    revisionHash=$(git --git-dir="${gitDir}" --work-tree="${src}" rev-parse HEAD)
    git --git-dir="${gitDir}" --work-tree="${src}" add -A
    git --git-dir="${gitDir}" --work-tree="${src}" diff --cached --no-color --no-ext-diff HEAD > "$revisionTmp/revision.diff"

    printf '%s' "$revisionHash" > revision.bin
    if [ -s "$revisionTmp/revision.diff" ]; then
      printf '\n\n' >> revision.bin
      cat "$revisionTmp/revision.diff" >> revision.bin
    fi

    revisionSize=$(wc -c < revision.bin)
    revisionBytes=$(od -An -v -tu1 revision.bin | awk '{for (i = 1; i <= NF; ++i) printf "%s,", $i}' | sed 's/,$//')
    printf '%s' '#include <cstddef>
extern const unsigned char turbogpt_revision[] = { '"$revisionBytes"' };
extern const size_t turbogpt_revision_size = '"$revisionSize"';
' > revision.cpp

    nvcc ${toString escapedFlags} \
      cuda/runtime.cu cuda/engine_state.cu checkpoint.cpp sampling.cpp train.cpp \
      trainer.cpp tensorboard.cpp training_log.cpp dataset.cpp loss_summary.cpp \
      model.cpp report.cpp revision.cpp \
      -L${cuda.cuda_cudart}/lib/stubs -lcuda -o turbogpt

    nvcc ${toString escapedFlags} -Xcompiler -fPIC -shared \
      cuda/runtime.cu cuda/engine_state.cu \
      -L${cuda.cuda_cudart}/lib/stubs -lcuda -o libturbogpt.so

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/bin" "$out/lib"
    install -Dm755 turbogpt "$out/bin/turbogpt"
    install -Dm755 libturbogpt.so "$out/lib/libturbogpt.so"

    runHook postInstall
  '';

  meta = with pkgs.lib; {
    description = "Tiny byte-level CUDA GPT with fused training";
    mainProgram = "turbogpt";
    license = licenses.mit;
    platforms = platforms.linux;
  };
}
