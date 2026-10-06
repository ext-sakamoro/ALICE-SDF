# はじめに

ALICE-SDF の紹介: この crate が何をするか、形状の語彙、形状を組み立て・評価・書き出す
一般的な方法 短い版は [README](../README_JP.md)、こちらは長い版

[English](GETTING_STARTED.md)

## 機能一覧
ALICE-SDFは、ポリゴンメッシュの代わりに**形状の数学的記述**（符号付き距離関数 = SDF）を伝送する3D/空間データスペシャリストです。これにより以下が実現されます:

- **10〜1000倍の圧縮** - 従来のメッシュフォーマットと比較
- **無限解像度** - あらゆるスケールで数学的に完全な形状
- **CSG演算** - メッシュオーバーヘッドなしの形状ブーリアン演算
- **計量を値として持つ** (4.0.0) - `MetricBall` はノルム自体を parameter にする (`‖p‖₂ − r` が球、`‖p‖∞ − r` が立方体、`‖p‖₁ − r` が八面体 = 式は同じでノルムだけが違う) Lipschitz 上界は推定でなく**閉形式** `MetricBlend` はバブルの内と外で別の場を使い、皮の外の世界を bit 一致で保つ `measure_tension` は領域上で場の勾配が実際に何をしているかを測る (2 つの法が出会う継ぎ目は、どちらの静的上界でも記述できない)
- **プラットフォーム横断の bit-exact 評価** (3.1.0) - 法則と評価器の超越関数を全て [`alice-det-math`](https://crates.io/crates/alice-det-math) (`alice-physics` と同じ crate) 経由にし `a * b + c` を fuse しないため、tree / compiled scalar / `f32x8` SIMD / BVH / Cranelift SIMD-JIT の各評価器が x86_64 / aarch64 / wasm32 で *同じ bit* を返す (`tests/test_det_parity.rs`、`tests/test_det_golden.rs`) GPU shader は tolerance 領域のまま、ただし `atan2` の軸上 tie は `alice_atan2` で CPU 法則に固定
- **リアルタイムレイマーチング** - GPU加速レンダリング
- **PBRマテリアル** - UE5/UE6/Unity/Godot互換のメタリック-ラフネスワークフロー
- **キーフレームアニメーション** - タイムライントラック付きパラメトリック変形
- **アセットパイプライン** - OBJ、glTF 2.0 (.glb)、FBX、USD、Alembic、Nanite、STL、PLY、3MF、ABM、Unity、UE5/UE6エクスポート
- **マニフォールドメッシュ保証** - バリデーション、修復、品質メトリクス
- **印刷可能性の検証** (`validity`) - 「これは印刷できるか」に**どう判定したかを添えて**答える 大域判定は `eval_interval` による erosion の証明で三値 `ErosionVerdict` (`HasThickEnoughRegion` / `EntirelyTooThin` / `Undecided`、**`Undecided` は合格ではない**)、局所肉厚は三角形ごとの sphere tracing で厳密実測、overhang は閉形式 `asin(-n · b)` `export_step_validated` は要件を満たさない形状の file を書かない
- **適応型マーチングキューブ** - オクツリーベースのメッシュ生成、必要な箇所にディテールを集中 (外向き CCW 巻き順 + 水密、1.11.0 で index 順を反転 — CHANGELOG 参照)
- **Dual Contouring** - QEFベースのメッシュ生成、シャープエッジとコーナーを保持
- **V-HACD凸分解** - 物理用自動凸包分解
- **属性保存デシメーション** - UV/タンジェント/マテリアル境界保護付きQEM
- **簡略化アドバンストオプション** - LODシーム保持用の頂点単位ロックマスク（`lock_vertices`）、絶対/相対誤差しきい値（`error_absolute`）
- **デシメーションベースLOD** - 高解像度ベースメッシュからのプログレッシブLODチェーン
- **meshopt互換コーデック** - バイナリ互換のインデックス/頂点バッファ圧縮（indexcodec v1 + vertexcodec v0/v1）、zeux/meshoptimizer C++ ライブラリが生成する参照ベクタで検証済
- **glTF `EXT_meshopt_compression`** - `io::meshopt_gltf` モジュール + `GltfConfig::meshopt_compress` オプション。POSITION / NORMAL / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0 / インデックスを属性別チャンネル推定（u8 / u16 / u32 XOR + 8回転ヒューリスティック）で圧縮
- **頂点フィルタ** - Octahedral（法線/タンジェント、角度誤差 1% 未満で 50〜75% 削減）、Quaternion（回転、最大成分＋循環スウィズル）、Exponential（浮動小数の仮数＋共有指数）
- **トライアングルストリップ化** - Evans-Skiena-Varshney 貪欲ストリップアルゴリズム（8トライアングル先読みバッファ、閉じたメッシュでインデックス数を約 48% 削減）、プリミティブリスタート／退化三角形結合の両モード対応
- **Naniteスタイル meshlet クラスタ** - V2 隣接情報 + `cone_weight` 拡張と `NormalCone`（basic + `cone_apex`）による Vulkan `VK_EXT_mesh_shader` / DirectX 12 mesh shader カリング対応
- **74プリミティブ、25演算、7トランスフォーム、24モディファイア**（`SdfNode` の 130 variant、`SdfNode::category()` で数えた値）
- **5層メッシュ永続化** - ABMバイナリフォーマット、LODチェーン永続化、FIFO排出チャンクキャッシュ、Unity/UE5/UE6ネイティブエクスポート
- **Chamfer & Stairsブレンド** - ハードエッジベベルおよびステップ状CSG遷移
- **区間演算（Interval Arithmetic）** - 空間プルーニング用の保守的AABB評価とリプシッツ定数追跡
- **緩和球トレーシング（Relaxed Sphere Tracing）** - オーバーリラクゼーション (Keinert 2014、overshoot 時にリトリート) + リプシッツ適応ステップ TPMS 系 (Gyroid / Neovius 等) は距離場でないため `RaymarchConfig::relaxed(&node)` が必須
- **ニューラルSDF** - 複雑シーンを~10-100倍高速に近似する純Rust MLP
- **SDF対SDFコリジョン** - 区間演算AABBプルーニング付きグリッドベース接触検出
- **CSGツリー最適化** - 恒等変換/モディファイア除去、ネスト変換マージ、Smooth→Standard降格
- **解析的勾配（Analytic Gradient）** - 連鎖律とヤコビアン伝播による単一パス勾配計算（9解析+44数値フォールバックプリミティブ）
- **自動微分（Automatic Differentiation）** - 双対数前方モードAD、ヘッシアン推定、平均曲率計算
- **2D SDFモジュール** - 純粋2Dプリミティブ（circle、rect、bezier、フォントグリフ）とバイリニアサンプリング
- **CSGツリーDiff/Patch** - アンドゥ/リドゥおよびネットワーク同期用のSDFツリー構造差分
- **パラメトリック拘束ソルバー** - 幾何拘束（固定、距離、和、比率）のガウス-ニュートン最適化
- **距離場ヒートマップ** - 4カラーマップ（coolwarm、binary、viridis、magma）による断面スライス
- **Shell / Offset Surface** - 内側/外側オフセット制御付き可変厚シェルモディファイア
- **体積・表面積** - 決定論的PRNGと標準誤差を用いたモンテカルロ推定
- **ALICE-Fontブリッジ** - フォントグリフ → 2D/3D SDF変換、テキストレイアウト、3D押し出し（`--features font` は alice-font が publish されるまで crates.io では inert な gate — [インストール](../README_JP.md#インストール) の注意書き参照、ブリッジは `git` dep で利用）
- **自動タイトAABB** - 区間演算＋二分探索によるSDF表面を含む最小バウンディングボックス計算
- **7つの評価モード** - インタプリタ、コンパイルVM、SIMD 8-wide、BVH、SoAバッチ、JIT、GPU
- **3つのシェーダーターゲット** - GLSL、WGSL、HLSLトランスパイル
- **エンジン統合** - Unity、Unreal Engine 5 / 6、VRChat、Godot、WebAssembly


## 2 つの入口: 本 crate の API と LOL 言語

ALICE-SDF は **評価器** です。法則 (距離関数)、compile 済 backend (scalar /
SIMD / BVH / JIT)、shader transpiler、mesh pipeline を持ちますが、その tree が
どう書かれたかには関与しません。

[**ALICE-LOL**](https://github.com/ext-sakamoro/ALICE-LOL) は **その法則を書く
ための言語** です。同じ `SdfNode` tree に parse される DSL に加えて、「この形は
制約を満たすか」を三値 (充足 / 違反 / **未決定**) で答える法則検証器を持ちます。
未決定が黙って合格に繰り上がることはありません。

同じ形を 2 通りで:

```rust
// A: 本 crate の builder API
use alice_sdf::prelude::*;
let a = SdfNode::sphere(1.0).subtract(SdfNode::box3d(1.0, 1.0, 1.0));

// B: LOL DSL を実行時に parse (LLM が出力するのはこちら)
use alice_lol::runtime_parser::parse_lol;
let b = parse_lol("subtract(sphere(1.0), box3d(0.5, 0.5, 0.5))").unwrap();

// 同じ場: どちらも alice_sdf::eval を通る
assert_eq!(eval(&a, Vec3::new(0.7, 0.2, 0.1)), alice_lol::eval(&b, Vec3::new(0.7, 0.2, 0.1)));
```

箱の引数に注意: **本 crate の `SdfNode::box3d` は全長**を取り (内部で半分に
する)、**LOL の `box3d` は半幅**を取ります (DSL は variant の field を直接
書く)。読み方を揃えたい場合は本 crate 側の `SdfNode::box3d_half_extents` を使います
(この 2 例が一致することは ALICE-LOL の `tests/readme_parity.rs` が固定しています)。

**LOL を選ぶ場面**: text を入れて geometry を出したい (LLM 生成、GBNF
constrained decoding、prompt から印刷可能な STL/3MF)、または法則検証器が
必要な時。**本 crate を直接使う場面**: Rust で tree を組み、評価器 / meshing /
shader 出力が目的の時。

## コア crate の噛み合い方

この 4 つは詰め合わせではなく 1 つの機構として作っている それぞれが担うものは 1 つだけで、
境目の設計がそのまま価値になっている

| Crate | 担うもの | 接点 |
|-------|---------|------|
| [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) | 言語と法則検証器 | ALICE-SDF の `SdfNode` に parse する 判定は三値 (合格 / 違反 / **未定**) で、未定を合格に昇格させない |
| [ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF) | 距離関数と全 backend (scalar / SIMD / BVH / JIT / shader transpiler / mesh) | LOL が書いた木を評価し、ALICE-Physics に collider を渡す |
| [ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics) | 128-bit 固定小数点の剛体 / CCD / XPBD | 描画しているのと同じ場に衝突する (別の近似を持たない) |
| [ALICE-DetMath](https://github.com/ext-sakamoro/ALICE-DetMath) | `sin` / `cos` / `atan2` 等を bit 単位で規定 | 接点そのもの ALICE-SDF と ALICE-Physics が platform libm でなくこれを呼ぶ |

噛み合わせている理由は 1 つ、**同じ入力がどの platform でも同じ bit を返すこと** 機械ごとに
食い違う場は、寸法どおり印刷することも、法則で検証することも、lockstep で再生することも
できない だから超越関数は crate ごとに再実装せず共有している

> `alice-det-math` は解決後の依存グラフで version を揃えること 1 つの木に 2 version 入ると
> 同じ関数の実装が 2 つ存在することになり、保証が消える 確認は `cargo tree -i alice-det-math`

## インストールの補足

> **crates.io でのブリッジ feature** — 1.12.0 以降 `physics` (alice-physics 1.1) / `codec` (alice-codec 0.1.2) / `asp` (libasp 1.0) / `sdf-cache` (alice-cache 0.2) は crates.io の隣接 crate に解決され、CI の `bridges` job で検証されています `font` は alice-font が publish されるまで inert な gate で、`font_bridge` module はローカルの `alice-font` path dep と `RUSTFLAGS="--cfg alice_font_bridge"` が必要です (1.7.7 〜 1.11.0 の間は 5 つとも `[features]` から外れていました、CHANGELOG の `[v1.7.7]` / `[v1.12.0]` 参照)

### Claude Code / Codex 向け skill

`skills/implicit-cad/` は ALICE-SDF を Claude Code / Codex 向けのインストール可能な skill としてまとめたもの SDF の記述、GLSL / WGSL / HLSL / MSL への変換、メッシュ出力 (GLB / OBJ / STL / PLY / 3MF) をこの crate の薄い CLI ラッパーとして提供する [`skills/implicit-cad/SKILL.md`](../skills/implicit-cad/SKILL.md) を参照 対になる `alice-lol-sdf` skill ([ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) repo) は、LLM の制約付きデコードに対応した DSL の入口を提供する

## コアコンセプト

### SDF（符号付き距離関数）

SDFは任意の点から表面までの最短距離を返します:
- **負** = 形状の内部
- **ゼロ** = 表面上
- **正** = 形状の外部

### SdfNodeツリー構造

```
SdfNode
  |-- プリミティブ (74): Sphere, Box3D, Cylinder, Torus, Plane, Capsule, Cone, Ellipsoid,
  |                    RoundedCone, Pyramid, Octahedron, HexPrism, Link, Triangle, Bezier,
  |                    RoundedBox, CappedCone, CappedTorus, InfiniteCylinder, RoundedCylinder,
  |                    TriangularPrism, CutSphere, CutHollowSphere, DeathStar, SolidAngle,
  |                    Rhombus, Horseshoe, Vesica, InfiniteCone, Heart, Gyroid, MetricBall,
  |                    Tube, Barrel, Diamond, ChamferedCube, SchwarzP, Superellipsoid, RoundedX,
  |                    Pie, Trapezoid, Parallelogram, Tunnel, UnevenCapsule, Egg,
  |                    ArcShape, Moon, CrossShape, BlobbyCross, ParabolaSegment,
  |                    RegularPolygon, StarPolygon, Stairs, Helix,
  |                    Tetrahedron, Dodecahedron, Icosahedron,                    ← プラトン立体 (GDF)
  |                    TruncatedOctahedron, TruncatedIcosahedron,                 ← アルキメデス立体
  |                    BoxFrame,                                                   ← IQワイヤーフレームボックス
  |                    DiamondSurface, Neovius, Lidinoid, IWP, FRD,              ← TPMS曲面
  |                    FischerKochS, PMY,                                          ← TPMS曲面
  |                    Circle2D, Rect2D, Segment2D, Polygon2D,                   ← 2Dプリミティブ（押し出し）
  |                    RoundedRect2D, Annular2D,                                   ← 2Dプリミティブ（押し出し）
  |                    Terrain                                                     ← 手続き的バイオーム地形（FBM + Voronoi侵食）
  |-- 演算 (25): Union, Intersection, Subtraction,
  |              SmoothUnion, SmoothIntersection, SmoothSubtraction,
  |              ChamferUnion, ChamferIntersection, ChamferSubtraction,
  |              StairsUnion, StairsIntersection, StairsSubtraction,
  |              ExpSmoothUnion, ExpSmoothIntersection, ExpSmoothSubtraction,     ← IQ指数スムース
  |              XOR, Morph,                                                       ← ブーリアン/補間
  |              ColumnsUnion, ColumnsIntersection, ColumnsSubtraction,            ← hg_sdfカラム
  |              Pipe, Engrave, Groove, Tongue,                                    ← hg_sdf高度操作
  |              MetricBlend                                                       ← 泡の内側と外側で別の場
  |-- トランスフォーム (7): Translate, Rotate, Scale, ScaleNonUniform,
  |                        ProjectiveTransform,                                    ← 逆行列付き射影変換
  |                        LatticeDeform,                                          ← 自由形状変形（FFD）グリッド
  |                        SdfSkinning                                             ← ボーンウェイトスケルタル変形
  |-- モディファイア (24): Twist, Bend, RepeatInfinite, RepeatFinite, Noise, Round, Onion, Elongate,
  |                   Mirror, Revolution, Extrude, Taper, Displacement, SineDisplacement, PolarRepeat, SweepBezier,
  |                   Shear,                                                       ← 3軸せん断変形
  |                   OctantMirror,                                                ← 48重対称性
  |                   IcosahedralSymmetry,                                         ← 120重正二十面体対称性
  |                   IFS,                                                         ← 反復関数系フラクタル
  |                   HeightmapDisplacement,                                       ← ハイトマップ駆動表面変位
  |                   SurfaceRoughness,                                            ← FBMノイズラフネス
  |                   Animated,                                                    ← タイムライン駆動パラメータアニメーション
  |                   WithMaterial                                                 ← PBRマテリアル割り当て
```

## 使い方

### パスを選ぶ

役割ごとに最適な入り口を選んでください。全パスは同じ `SdfNode` 中間表現を共有するので、混ぜて使えます (例: LOL で組んで GLSL に transpile、Python で評価)。

| あなたは… | パス | 用途 | セクション |
|-----------|------|-----|-----------|
| **Rust 開発者、宣言的に書きたい** | [ALICE-LOL DSL](#alice-lol-dsl-で書く推奨) | シーン構築、GPU shader transpile、コンパイル時 law チェック | ↓ |
| **Rust 開発者、低レベル制御が要る** | [Rust 直接構築](#rust直接-sdfnode-を構築する場合) | カスタム modifier ノード、手動最適化 | ↓ |
| **Python / データサイエンス** | [Python バインディング](#python) | NumPy バッチ評価、メッシュ export、ノートブック運用 | ↓ |
| **Unity / UE5 / Godot 統合** | C-ABI FFI | ネイティブプラグイン、ゲームエンジンからのリアルタイム評価 | [docs/UNREAL_ENGINE.md](UNREAL_ENGINE.md) / [docs/GODOT_GUIDE.md](GODOT_GUIDE.md) |
| **Web / WebGPU 開発者** | WASM build | ブラウザ側 SDF 評価 + WGSL shader コンパイル | [docs/WASM_GUIDE.md](WASM_GUIDE.md) |
| **モバイル (iOS / Android)** | XCFramework / AAR | Swift / Kotlin でのデバイス上評価 | [Mobile セクション](INTEGRATIONS.md#mobile-ios--android) |
| **3D アーティスト / VFX** | Cookbook レシピ | 手続き形状 / displacement / タイリングをコピペ | [docs/VFX_COOKBOOK.md](VFX_COOKBOOK.md) |
| **初めて触る** | 下の 30 秒サンプル ↓ | インストール確認 | ↓ |

### Hello, First SDF (30 秒)

最小の実用サンプル — 構築、評価、メッシュ化、完了:

```rust
use alice_sdf::prelude::*;

let sphere = SdfNode::sphere(1.0);
let d = eval(&sphere, glam::Vec3::new(0.5, 0.0, 0.0));
assert!((d + 0.5).abs() < 1e-6);           // 中心から 0.5 の点 → distance -0.5 (内側)

let mesh = sdf_to_mesh(
    &sphere,
    glam::Vec3::splat(-1.5),
    glam::Vec3::splat(1.5),
    &MarchingCubesConfig::default(),
);
println!("{} 頂点、{} 三角形", mesh.vertices.len(), mesh.indices.len() / 3);
```

これが動いたら、上の表で自分の役割に合ったパスに進んでください。

### ALICE-LOL DSL で書く（推奨）

SDF シーンを作る最も簡単な方法は [ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL) です。`lol!` proc_macro で SDF ツリーを宣言的に記述でき、手動で `SdfNode` を組み立てる必要がありません。

```toml
# Cargo.toml
[dependencies]
alice-sdf = { path = "../ALICE-SDF" }
alice-lol = { path = "../ALICE-LOL/alice-lol" }
```

**従来（手動で SdfNode を構築）:**

```rust
use alice_sdf::prelude::*;

let scene = SdfNode::SmoothUnion {
    k: 0.3,
    children: vec![
        SdfNode::sphere(1.0),
        SdfNode::Translate {
            offset: glam::Vec3::new(2.0, 0.0, 0.0),
            child: Box::new(SdfNode::Round {
                radius: 0.05,
                child: Box::new(SdfNode::box3d(0.8, 0.8, 0.8)),
            }),
        },
    ],
};
```

**LOL DSL で書くと:**

```rust
use alice_lol::{lol, to_glsl, eval};

let scene = lol! {
    smooth_union(0.3,
        sphere(1.0),
        translate(2.0, 0.0, 0.0, round(0.05, box3d(0.8, 0.8, 0.8)))
    )
};
```

同じ `SdfNode` ツリーが、わずかなコードで完成します。76 構文（27 プリミティブ、23 CSG オペレーション、4 トランスフォーム、19 モディファイア、2 時間制御、3 法則制約）がすべて関数呼び出しで使えます。

**GPU シェーダにトランスパイル:**

```rust
let glsl = to_glsl(&scene);                      // GLSL
let wgsl = alice_lol::to_wgsl(&scene);            // WGSL (WebGPU)
let hlsl = alice_lol::to_hlsl(&scene);            // HLSL (DirectX)
```

**CPU で距離を評価:**

```rust
let dist = eval(&scene, glam::Vec3::new(0.0, 1.0, 0.0));
```

**LOL シーンからメッシュ書き出し:**

```rust
use alice_lol::lol;
use alice_sdf::prelude::*;

let scene = lol! {
    smooth_union(0.3,
        sphere(1.0),
        translate(2.0, 0.0, 0.0, box3d(0.8, 0.8, 0.8))
    )
};

let mesh = sdf_to_mesh(
    &scene,
    glam::Vec3::splat(-3.0),
    glam::Vec3::splat(3.0),
    &MarchingCubesConfig { resolution: 128, ..Default::default() },
);

alice_sdf::export::obj::write_obj("out.obj", &mesh)?;
alice_sdf::export::glb::write_glb("out.glb", &mesh)?;
```

**Rust の変数を実行時に注入:**

```rust
let radius = 1.5_f32;
let height = compute_height();
let scene = lol! {
    smooth_union(0.2,
        sphere({radius}),
        translate(0.0, {height}, 0.0, cylinder(2.0, 0.5))
    )
};
```

**形状の制約をチェック:**

```rust
use alice_lol::law::{LawSet, Law, Priority};

let laws = LawSet::new()
    .add(Law::non_overlap(&a, &b), Priority::Hard)        // 形状が重ならないこと
    .add(Law::min_thickness(&scene, 0.1), Priority::Soft(0.5));  // 壁厚 >= 0.1
let report = laws.check();
```

LOL の詳細は [ALICE-LOL README](https://github.com/ext-sakamoro/ALICE-LOL) を参照してください。

---

### Rust（直接 SdfNode を構築する場合）

LOL DSL でカバーされていない高度なノード型を使う場合や、細かい制御が必要な場合は `SdfNode` を直接構築できます:

```rust
use alice_sdf::prelude::*;

// 半径1の球体を作成
let sphere = SdfNode::sphere(1.0);

// 箱でくり抜く
let result = sphere.subtract(SdfNode::box3d(1.5, 1.5, 1.5));

// ある点での距離を評価
let distance = eval(&result, glam::Vec3::ZERO);

// メッシュに変換
let mesh = sdf_to_mesh(
    &result,
    glam::Vec3::splat(-2.0),
    glam::Vec3::splat(2.0),
    &MarchingCubesConfig::default()
);
```

### Python

```python
import alice_sdf as sdf

# プリミティブを作成
sphere = sdf.SdfNode.sphere(1.0)
box3d = sdf.SdfNode.box3d(2.0, 1.0, 1.0)

# CSG演算（メソッド構文）
result = sphere.subtract(box3d)

# 演算子オーバーロード（Python的な構文）
a = sdf.SdfNode.sphere(1.0)
b = sdf.SdfNode.box3d(0.5, 0.5, 0.5)
union     = a | b    # a.union(b)
intersect = a & b    # a.intersection(b)
subtract  = a - b    # a.subtract(b)

# トランスフォーム
translated = result.translate(1.0, 0.0, 0.0)

# 点群で評価（NumPy配列）
import numpy as np
points = np.array([[0.5, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
distances = sdf.eval_batch(translated, points)

# コンパイル評価（繰り返し使用時2-5倍高速）
compiled = sdf.compile_sdf(sphere)
distances = compiled.eval_batch(points)               # コンパイルバッチ
vertices, indices = compiled.to_mesh((-2,-2,-2), (2,2,2), resolution=64)  # コンパイルメッシュ

# メッシュに変換
vertices, indices = sdf.to_mesh(translated, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))

# 複数フォーマットにエクスポート
sdf.export_obj(vertices, indices, "model.obj")
sdf.export_glb(vertices, indices, "model.glb")
sdf.export_fbx(vertices, indices, "model.fbx")
sdf.export_usda(vertices, indices, "model.usda")
sdf.export_alembic(vertices, indices, "model.abc")

# UV展開 → (positions[N,3], uvs[N,2], indices[M])
positions, uvs, indices = sdf.uv_unwrap(vertices, indices)
```


### Sine Displacement (v1.7.3、等方 + 異方対応)

`SineDisplacement` は任意の子 SDF に sin 波の摂動を加える modifier。**v1.7.3 で per-axis `Vec3` frequency に対応**、XYZ 独立の細かいパターン (鱗、木目、布のドレープ) を 1 ノードで表現できる。

2 種類のコンストラクタ:

| メソッド | frequency 型 | 用途 |
|----------|------------|-----|
| `.sine_displacement(amplitude, freq: f32)` | 等方 (`Vec3::splat`) | 均一な波紋 / 表面荒れ / 細胞状の鱗 |
| `.sine_displacement_aniso(amplitude, freq: Vec3)` | 軸ごと | 木目、布ドレープ、細長い鱗、単軸に沿う波面 |

```rust
use alice_sdf::prelude::*;
use glam::Vec3;

// 等方: 球体に細かい細胞状の摂動
let scales = SdfNode::sphere(1.0).sine_displacement(0.03, 25.0);

// 異方: 細長い鱗 (X 方向高周波、Y/Z 低周波)
let fish_scales = SdfNode::sphere(1.0).sine_displacement_aniso(0.03, Vec3::new(40.0, 10.0, 10.0));

// 異方: 木目 (Y 方向密、他は疎)
let wood = SdfNode::box3d(1.0, 3.0, 1.0)
    .sine_displacement_aniso(0.01, Vec3::new(4.0, 30.0, 4.0));
```

両 variant とも標準パイプライン (`to_glsl` / `to_wgsl` / `to_hlsl`) で GLSL / WGSL / HLSL に transpile 済み。transpiler は per-axis `sin(freq.x * p.x) * sin(freq.y * p.y) * sin(freq.z * p.z)` を吐き、CPU 側 `modifier_sine_displacement` reference と一致。**区間演算バウンドと勾配 (diff) サポートも既に配線済み**なので raymarching と勾配ベースサンプリングで安全に使える。

### よく使うレシピ

繰り返し出てくるパターンをコピペしやすい形で。全例で `use alice_sdf::prelude::*;` 前提。

**1. 角丸ボックス**

```rust
let rounded = SdfNode::box3d(1.0, 1.0, 1.0).round(0.15);
```

**2. スムーズ blob (ソフトブレンド Union)**

```rust
let blob = SdfNode::sphere(1.0)
    .smooth_union(SdfNode::sphere(0.7).translate(1.2, 0.0, 0.0), 0.4);
// signature: smooth_union(self, other: Self, k: f32) — k はブレンド半径
```

**3. 中空シェル (onion)**

```rust
let shell = SdfNode::sphere(1.0).onion(0.05);   // 壁厚 5cm
```

**4. 無限タイリング (`repeat_infinite`)**

```rust
let tiles = SdfNode::box3d(0.4, 0.4, 0.4).repeat_infinite(1.0, 1.0, 1.0);
// 有限版: .repeat_finite([count_x, count_y, count_z], spacing)
```

**5. Y 軸ねじり**

```rust
let twisted = SdfNode::box3d(0.4, 2.0, 0.4).twist(1.5);   // Y 周り 1.5 rad/unit
```

**6. Displacement (sin + noise)**

```rust
let rough = SdfNode::sphere(1.0).sine_displacement(0.03, 20.0);
```

**7. CSG 連鎖 (プレートに穴を空ける)**

```rust
let plate = SdfNode::box3d(2.0, 0.1, 2.0);
let hole  = SdfNode::cylinder(2.0, 0.15);
let drilled = plate
    .subtract(hole.translate(-0.8, 0.0,  0.0))
    .subtract(hole.translate( 0.8, 0.0,  0.0))
    .subtract(hole.translate( 0.0, 0.0, -0.8));
```

**8. GPU transpile (WebGPU / Metal / DX12)**

```rust
use alice_lol::{lol, to_wgsl};

let scene = lol! { smooth_union(0.3, sphere(1.0), box3d(0.8, 0.8, 0.8)) };
let wgsl_source = to_wgsl(&scene);      // WGSL shader に貼り付け
```

**VFX 系パターン** (流体、mandelbulb、魔法エフェクト、リボン FX、プラズマ、ポータル、force field) は [`docs/VFX_COOKBOOK.md`](VFX_COOKBOOK.md) を参照。

**汎用レシピ集** (プロシージャル地形、カメラ相対 modifier、バウンディング volume trick、LOD 戦略) は [`docs/COOKBOOK.md`](COOKBOOK.md) を参照。

### 次に見るべきドキュメント

| やりたいこと | 参照先 |
|------------|-------|
| 全 `SdfNode` variant を把握 | [`docs/API_REFERENCE.md`](API_REFERENCE.md) |
| 新規プロジェクト立ち上げ | [`docs/QUICKSTART.md`](QUICKSTART.md) |
| コンパイラ / evaluator 内部を深掘り | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) |
| Unity / Unreal / Godot 統合 | [`docs/UNREAL_ENGINE.md`](UNREAL_ENGINE.md) · [`docs/GODOT_GUIDE.md`](GODOT_GUIDE.md) |
| Python から使う | [`docs/PYTHON_GUIDE.md`](PYTHON_GUIDE.md) |
| ブラウザで動かす | [`docs/WASM_GUIDE.md`](WASM_GUIDE.md) |
| 3D プリント用パーツを作る | [ALICE-Bamboo](https://github.com/ext-sakamoro/ALICE-Bamboo) 参照 (LOL → SDF → 3MF パイプライン) |

詳細な技術セクション (マテリアル / アニメーション / アーキテクチャ / メッシュモジュール / プラトン立体 / 区間演算 / ニューラル SDF / コリジョン / 解析的勾配 / Dual Contouring / CSG最適化 / 自動タイト AABB / テクスチャフィッティング / レイマーチング / FFI / フィーチャーフラグ / 物理ブリッジ / 3D プリントパイプライン / パフォーマンス / ベンチマーク / Unity / VRChat / UE5・UE6 / Godot / クロスクレートブリッジ / Asset Delivery Network / Nanite ハイブリッドパイプライン) は [`docs/USAGE_JP.md`](USAGE_JP.md) を参照。

