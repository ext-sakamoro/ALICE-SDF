# ALICE-SDF

<p align="center">
  <img src="asset/logo-on-light.jpeg" alt="ALICE-SDF Logo" width="480">
</p>

Rust の符号付き距離関数 (SDF) ライブラリ 形状はプリミティブ・CSG 演算・変換・
モディファイアの木で表す この crate は木を CPU で評価し (スカラー / SIMD / BVH /
JIT)、GLSL・WGSL・HLSL・Metal に変換し、メッシュや各種ファイル形式に書き出す

[English](README.md) | 日本語

[![crates.io](https://img.shields.io/crates/v/alice-sdf.svg)](https://crates.io/crates/alice-sdf)
[![docs.rs](https://img.shields.io/docsrs/alice-sdf)](https://docs.rs/alice-sdf)
[![MSRV](https://img.shields.io/crates/msrv/alice-sdf)](#最小対応-rust-バージョン)
[![CI](https://github.com/ext-sakamoro/ALICE-SDF/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-SDF/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/alice-sdf.svg)](#ライセンス)

同じ木がすべてのバックエンドを駆動し、CPU の評価器どうしは CI が検証する全
プラットフォームで同じビット列を返す そのため木を形状の唯一の記述として使える
描画するシェーダー、物理エンジンが問い合わせるコライダー、書き出し・印刷する
メッシュはすべて同じ木から計算され、別々の近似を持たない
<!-- claim-test: every_cpu_evaluator_is_bit_identical -->

メッシュモデラーやレンダラーではない GPU シェーダーは CPU とビット一致せず
(許容誤差内で一致)、一部のモディファイア (twist / bend / displacement など) を
通した値は距離そのものではなく距離の上界になる また場を細かいメッシュにする
コストは解像度に応じて時間・メモリともに増える

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [決定性](#決定性)
- [含まれるもの](#含まれるもの)
- [検証状況と既知の不具合](#検証状況と既知の不具合)
- [Cargo feature](#cargo-feature)
- [バインディング](#バインディング)
- [性能](#性能)
- [最小対応 Rust バージョン](#最小対応-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [関連 crate](#関連-crate)
- [ライセンス](#ライセンス)

## インストール

```sh
cargo add alice-sdf
```

コマンドラインツール (既定で有効な `cli` feature) を除く場合:

```sh
cargo add alice-sdf --no-default-features
```

Python バインディングは PyPI にある:

```sh
pip install alice-sdf
```

## 使用例

球から箱をくり抜き、評価・コンパイル・メッシュ化する 同じコードが `src/lib.rs`
の crate レベル doctest なので、`cargo test` でコンパイル・実行される

```rust
use alice_sdf::prelude::*;

// A unit sphere minus a box (box3d takes full extents)
let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(1.0, 1.0, 1.0));

// Signed distance: negative inside, zero on the surface, positive outside
let p = Vec3::new(0.9, 0.0, 0.0);
let d = eval(&shape, p);
assert!(d < 0.0);

// Compiled to bytecode for repeated evaluation, with the same bits
let compiled = CompiledSdf::compile(&shape);
assert_eq!(eval_compiled(&compiled, p).to_bits(), d.to_bits());

// A triangle mesh by marching cubes
let mesh = sdf_to_mesh(
    &shape,
    Vec3::splat(-1.5),
    Vec3::splat(1.5),
    &MarchingCubesConfig::default(),
);
assert!(!mesh.indices.is_empty());
```

ほかのプログラムは [`examples/`](examples/) に、長めの解説 (レシピ、Python、
LOL 言語) は [`docs/GETTING_STARTED_JP.md`](docs/GETTING_STARTED_JP.md) にある

## 決定性

CPU の評価器 (木の評価器、コンパイル済みスカラー、8 幅 SIMD、BVH、Cranelift JIT)
は、同じ木と同じ点に対してビット一致した距離を返す プラットフォームをまたいで
これが成り立つのは次の 2 つの規則による

- 超越関数 (`sin` / `atan2` / `exp` / `powf` など) はすべて
  [`alice-det-math`](https://crates.io/crates/alice-det-math) を通し、
  プラットフォームの `libm` は使わない
- `a * b + c` は常に 2 回丸める 積和演算 (FMA) に融合しない

`scripts/det_math_guard.py` は、評価器と法則のディレクトリにプラットフォームの
`libm` 呼び出しや `mul_add` があると CI を失敗させる

| テスト | 固定するもの | CI での実行環境 |
|------|--------------|---------------------|
| `tests/test_det_parity.rs` | すべての CPU 評価器を木の評価器とビット単位で比較 | macOS (ARM64)、Linux (x86_64)、Windows (x86_64)、JIT は `--features jit` |
| `tests/test_det_golden.rs` | コーパスの形状ごとに木の評価器のビット列の SHA-256 | 同じ 3 環境 |
| `tests/test_gpu_law_parity.rs` | WGSL を CPU と許容誤差内で比較、`atan2` の軸上の値は厳密に一致 | Linux、ソフトウェア Vulkan (lavapipe) |
<!-- claim-test: tree_evaluator_bits_match_recorded_hashes -->

**対象外** `wasm32` は CI でビルドするが、上のテストは実行していない GPU
シェーダー (WGSL / GLSL / HLSL / Metal) は許容誤差の領域で、ビット一致の対象では
ない 基本演算が IEEE 754 に従わないターゲットや、fast-math 系のフラグを付けた
ビルドは保証の外

**決定性は正しさではない** ビットが一致するのは全機械が同じ数値を計算すると
いうことで、数値が正しいかは別に検証している
[検証状況と既知の不具合](#検証状況と既知の不具合) を参照

## 含まれるもの

公開モジュールの全件 (分野別、1 行の説明と必要な feature) は
[`docs/MODULES.md`](docs/MODULES.md) にある API の詳細は
[docs.rs](https://docs.rs/alice-sdf) を参照

| 分野 | 主な内容 |
|------|-----------|
| 形状 | プラトン立体・アルキメデス立体・TPMS 曲面・押し出した 2D 形状を含むプリミティブ、ノルムを引数に取る `MetricBall` |
| 演算 | 和・積・差と、その smooth / chamfer / stairs / exponential / column 版、XOR、morph、`MetricBlend` |
| 変換とモディファイア | 平行移動・回転・拡大縮小・射影・格子変形・スキニング、twist・bend・繰り返し・鏡映と対称折り返し・displacement・shell・IFS |
| 評価 | 木の評価器、コンパイル済みバイトコード (スカラー / SIMD / BVH)、Cranelift JIT、wgpu による GPU 計算、区間演算、解析的・自動微分の勾配、リプシッツ上界 |
| シェーダー出力 | GLSL、WGSL、HLSL、Metal (WGSL 出力から naga で変換)、BlinkScript、依存のない Rust ソース |
| メッシュ化 | marching cubes (一様 / 適応 / GPU)、dual contouring、間引き、LOD チェーン、多様体修復、UV 展開、meshlet、meshopt 互換コーデック |
| ファイル形式 | `.asdf` / `.asdf.json` の木、OBJ、glTF、FBX、USD、Alembic、STL、PLY、3MF、STEP、IGES、MagicaVoxel、Gaussian splat、OpenVDB |
| 解析 | 印刷可能性 (3 値の判定を返す侵食の証明、肉厚、オーバーハング)、体積と表面積、タイトな境界箱、SDF 同士の衝突 |
| ワールド | スパースボクセル八分木、ボクセル破壊、侵食付き地形、コーントレースによる大域照明 |

## 検証状況と既知の不具合

`tests/` のテストは、結果を閉形式の値または独立な計算と比較する: smooth 演算、
メトリック場、区間の包含 (`test_interval_soundness.rs`)、タイトな境界、
印刷可能性、メッシュの向きと位相、STEP 出力、参照ライブラリのベクタに対する
meshopt コーデック、ファイル形式の往復 シェーダー出力は CI で naga (と Metal) で
コンパイルする ゴールデンハッシュは変化を検出するだけなので、これらとは分けている

[`docs/oracle-status.md`](docs/oracle-status.md) は `tests/` から生成され、全テストを
状態別に一覧する 実装が直るまで意図的に赤のまま置くテスト
(`#[ignore = "known defect: …"]`) もここに載る

[`docs/wiring-status.md`](docs/wiring-status.md) は、テスト以外のどこからも呼ばれない
公開アイテムの一覧 `scripts/wiring_guard.py` は、理由のない新規のものが現れると
CI を失敗させる

`scripts/ci_test_coverage_check.py` は、feature で切り替わるテストファイルをその
feature 付きで実行する CI ステップが無いと CI を失敗させる feature 付きのテストが
0 件実行のまま green になる経路を塞ぐため

## Cargo feature

**AGPL** と記した feature は `AGPL-3.0-or-later` の crate をリンクする
[ライセンス](#ライセンス) を参照 既定を含むそれ以外の feature は、寛容なライセンスの
crate だけを取り込む

<!-- readme-sync: features -->
| Feature | 既定 | 説明 |
|---------|:-------:|-------------|
| `cli` | yes | `alice-sdf` コマンドラインツール (clap) |
| `image` | | ハイトマップとテクスチャフィット用の PNG / JPEG デコード |
| `texture-fit` | | ビットマップテクスチャを手続き的なノイズ式で近似する `image` と `cli` を含む |
| `jit` | | Cranelift によるネイティブ評価 (スカラーと 8 幅) |
| `gpu` | | wgpu による GPU 評価と WGSL 出力 |
| `gpu-mesh` | | GPU 上の marching cubes `gpu` を含む |
| `volume` | | 場を 3D テクスチャに焼き込む `gpu` を含む |
| `glsl` | | GLSL 出力 (Unity / OpenGL / Vulkan / Shadertoy) |
| `hlsl` | | HLSL 出力 (Unreal Engine / DirectX) |
| `msl` | | Metal Shading Language 出力 (WGSL から naga で変換) `gpu` を含む |
| `blinkscript` | | Nuke 用 BlinkScript 出力 `hlsl` を含む |
| `rust` | | `eval_compiled` とビット一致する依存のない Rust ソース (`fn sdf` / `fn normal`)、`build.rs` 向け |
| `all-shaders` | | `gpu`、`glsl`、`hlsl`、`msl`、`blinkscript` |
| `svo` | | スパースボクセル八分木 |
| `svo-gpu` | | GPU 対応のスパースボクセル八分木 `svo` と `gpu` を含む |
| `destruction` | | ボクセル破壊 |
| `terrain` | | 侵食と洞窟を持つハイトマップ地形 |
| `gi` | | コーントレースによる大域照明 `svo` を含む |
| `aaa` | | `volume`、`gpu-mesh`、`svo-gpu`、`destruction`、`terrain`、`gi` |
| `ffi` | | C / C++ / C# / Unity / Unreal Engine 向けの C ABI |
| `unity` | | `ffi` と `glsl` |
| `unreal` | | `ffi`、`hlsl`、`glsl`、`gpu` (Unreal Engine プラグインが呼ぶもの) |
| `python` | | Python バインディング (PyO3 + NumPy) |
| `godot` | | Godot 4 GDExtension |
| `wasm` | | `wasm-bindgen` による WebAssembly バインディング (`wasm32` 向けのみビルド) |
| `openvdb` | | OpenVDB の float grid の入出力 |
| `physics` | | **AGPL** [`alice-physics`](https://crates.io/crates/alice-physics) 向けの SDF コライダーとシミュレーションモディファイア |
| `codec` | | **AGPL** [`alice-codec`](https://crates.io/crates/alice-codec) によるボリューム圧縮 |
| `sdf-cache` | | **AGPL** [`alice-cache`](https://crates.io/crates/alice-cache) による評価キャッシュ |
| `asp` | | [`libasp`](https://crates.io/crates/libasp) による ALICE Streaming Protocol パケット (既定 feature のみ) |
| `font` | | `alice-font` のグリフ輪郭 crates.io 版では無効: `--cfg alice_font_bridge` とローカルの `alice-font` も必要 |

## バインディング

| 対象 | 場所 | 備考 |
|--------|-------|-------|
| C / C++ | [`include/alice_sdf.h`](include/alice_sdf.h) | `--features ffi` |
| C# / Unity | [`bindings/AliceSdf.cs`](bindings/AliceSdf.cs)、[`unity-sdf-universe/`](unity-sdf-universe/README.md) | C ABI 上の P/Invoke |
| Unreal Engine 5 / 6 | [`unreal-plugin/`](unreal-plugin/README.md) | `--features unreal` |
| VRChat | [`vrchat-package/`](vrchat-package/README_JP.md) | プレイヤーが歩き、衝突できる SDF 表面の VRChat パッケージ |
| Godot 4 | [`docs/GODOT_GUIDE.md`](docs/GODOT_GUIDE.md) | `--features godot` |
| Python | [`python/`](python/)、[`docs/PYTHON_GUIDE.md`](docs/PYTHON_GUIDE.md) | `--features python`、NumPy による一括評価とメッシュ出力 |
| WebAssembly / Three.js | [`docs/WASM_GUIDE.md`](docs/WASM_GUIDE.md)、[`npm/`](npm/README.md) | `--features wasm` |
| iOS / Android | [`mobile/`](mobile/README.md) | C ABI 上の XCFramework と Kotlin バインディング |
| Bevy / Blender / Houdini / Maya / Nuke / Cinema 4D / OpenXR / visionOS | [`bindings/`](bindings/README.md)、[`docs/INTEGRATIONS_JP.md`](docs/INTEGRATIONS_JP.md) | 参照実装 |

CI は C ABI をその利用側と突き合わせる: `scripts/abi_decl_check.py` と
`scripts/unreal-abi-check.sh` が、公開している全関数を C ヘッダー、C# バインディング、
Unreal Engine プラグインと比較する

Text-to-3D サーバーと ALICE-View ビューアは crate の上に作ったアプリケーション
[`docs/TEXT_TO_3D_JP.md`](docs/TEXT_TO_3D_JP.md) を参照

## 性能

ベンチマークは [`benches/`](benches/) にある (criterion):

```sh
cargo bench --bench sdf_eval
cargo bench --bench gpu_vs_cpu --features gpu
```

ここには数値を載せない 古い文書の数値は評価器をビット一致にする (FMA を外した)
前に測ったもので、その後測り直していない 数値を引用する前に、対象の環境で
ベンチマークを実行すること

## 最小対応 Rust バージョン

最小対応 Rust バージョン: **1.85** (`Cargo.toml` の `rust-version`) <!-- readme-sync: msrv -->

CI のジョブが、ちょうどこのバージョンで既定 feature と docs.rs の feature 組で
ライブラリを検査する MSRV の引き上げはマイナーバージョンの変更として扱い、
パッチでは上げない

## ビルドとテスト

```sh
cargo build --release
cargo test
cargo test --features jit --test test_det_parity
cargo test --features "gpu,glsl,gpu-mesh,texture-fit"
```

`scripts/preflight.sh` は CI の検査をローカルで再現する (`--quick` は統合テストと
オラクルテストを飛ばす)

## 関連 crate

| Crate | 役割 |
|-------|------|
| [alice-det-math](https://github.com/ext-sakamoro/ALICE-DetMath) | この crate と ALICE-Physics が共に使う決定的な超越関数 |
| [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) | 同じ `SdfNode` の木にパースされる言語と法則検証器 ([はじめに](docs/GETTING_STARTED_JP.md#2-つの入口-本-crate-の-api-と-lol-言語) を参照) |
| [ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics) | 決定的な物理エンジン `physics` feature で、描画しているのと同じ場に衝突する |
| [ALICE-View](https://github.com/ext-sakamoro/ALICE-View) | `.asdf` ファイルのリアルタイムビューア |

依存グラフの `alice-det-math` は 1 つの版にそろえること 2 つの版があると同じ関数の
実装が 2 つになり、決定性が失われる `cargo tree -i alice-det-math` で確認できる

リリース履歴は [`CHANGELOG.md`](CHANGELOG.md)、予定は
[`docs/ROADMAP.md`](docs/ROADMAP.md) にある

## ライセンス

crate は `MIT OR Apache-2.0` のデュアルライセンス
([LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE))

`physics`、`codec`、`sdf-cache` の各 feature は `AGPL-3.0-or-later` の crate を
リンクする いずれかを有効にしたビルドは全体が AGPL の対象になる 既定の feature と
それ以外の feature は対象にならない

Unity 統合 ([`unity-sdf-universe/`](unity-sdf-universe/)) と VRChat パッケージ
([`vrchat-package/`](vrchat-package/)) は ALICE Community License
([LICENSE-COMMUNITY](LICENSE-COMMUNITY)): 個人利用、ゲーム開発 (商用ゲームを含む)、
教育、オープンソースでは無償 同ライセンスが定めるインフラサービス (クラウド、
メタバースプラットフォーム、ストリーミング) には商用ライセンスが必要 商用
ライセンスの問い合わせ: <contact@extoria.co.jp>

この crate で作ったコンテンツ (木、メッシュ、ワールド) はあなたのもの

多くの距離関数は Inigo Quilez、Mercury (hg_sdf)、Ken Perlin が公開した式に従う
一覧と範囲は [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) にある

Copyright (C) 2025-2026 Moroya Sakamoto
