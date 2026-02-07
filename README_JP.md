# ALICE-SDF

[English](README.md) | **日本語**

**A.L.I.C.E. - Adaptive Lightweight Implicit Compression Engine**

> "ポリゴンを送るな。形の法則を送れ。"

## 概要

ALICE-SDFは、ポリゴンメッシュの代わりに**形状の数学的記述**（符号付き距離関数 = SDF）を伝送する3D/空間データスペシャリストです。これにより以下が実現されます:

- **10〜1000倍の圧縮** - 従来のメッシュフォーマットと比較
- **無限解像度** - あらゆるスケールで数学的に完全な形状
- **CSG演算** - メッシュオーバーヘッドなしの形状ブーリアン演算
- **リアルタイムレイマーチング** - GPU加速レンダリング
- **PBRマテリアル** - UE5/Unity/Godot互換のメタリック-ラフネスワークフロー
- **キーフレームアニメーション** - タイムライントラック付きパラメトリック変形
- **アセットパイプライン** - OBJインポート/エクスポート、glTF 2.0 (.glb)、FBX、USD、Alembic、Naniteエクスポート
- **マニフォールドメッシュ保証** - バリデーション、修復、品質メトリクス
- **適応型マーチングキューブ** - オクツリーベースのメッシュ生成、必要な箇所にディテールを集中
- **V-HACD凸分解** - 物理用自動凸包分解
- **属性保存デシメーション** - UV/タンジェント/マテリアル境界保護付きQEM
- **デシメーションベースLOD** - 高解像度ベースメッシュからのプログレッシブLODチェーン
- **53プリミティブ、4トランスフォーム、15モディファイア** - 豊富なシェイプボキャブラリ
- **7つの評価モード** - インタプリタ、コンパイルVM、SIMD 8-wide、BVH、SoAバッチ、JIT、GPU
- **3つのシェーダーターゲット** - GLSL、WGSL、HLSLトランスパイル
- **エンジン統合** - Unity、Unreal Engine 5、VRChat、Godot、WebAssembly

## Text-to-3D パイプライン（サーバー）

ALICE-SDFには、LLM生成のSDFツリーを通じて**自然言語テキストを実際の3Dジオメトリに変換する**FastAPIサーバーが含まれています。

```
ユーザー: "中世の城"  →  LLM (Claude/Gemini)  →  SDF JSON  →  ALICE-SDF  →  GLB/OBJ
         テキスト           ~5-50秒              20ノード      <55ms        メッシュ
```

### アーキテクチャ

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  FastAPI     │     │  LLMサービス │     │  SDFサービス  │     │  出力    │
│  サーバー    │────▶│  Claude API  │────▶│  from_json()  │────▶│  GLB     │
│              │     │  Gemini API  │     │  compile()    │     │  OBJ     │
│  POST /gen   │     │  システム    │     │  to_mesh()    │     │  JSON    │
│  WS /ws/gen  │     │  プロンプト  │     │  export_glb() │     │  ビューア│
│  GET /viewer │     └──────────────┘     └───────────────┘     └──────────┘
└─────────────┘
```

### APIエンドポイント

| メソッド | パス | 説明 |
|--------|------|-------------|
| `POST` | `/api/generate` | テキスト → 3Dメッシュ (GLB/OBJ/JSON) |
| `POST` | `/api/validate` | SDF JSON構造のバリデーション |
| `POST` | `/api/mesh` | SDF JSON → メッシュ (GLB/OBJ) |
| `GET` | `/api/examples` | Few-shotサンプルシーン一覧 |
| `GET` | `/api/viewer` | Three.js GLBビューア（ブラウザ） |
| `GET` | `/api/health` | サーバーヘルスチェック |
| `WS` | `/ws/generate` | プログレッシブプレビュー付きストリーミング生成 |

### 生成シーンギャラリー

Gemini 2.5 Flashが自然言語プロンプトから生成したシーン:

| プロンプト | ノード数 | 頂点数 | 三角形数 | LLM時間 |
|--------|-------|----------|-----------|----------|
| "A medieval castle with towers" | 18 | 2,105 | 4,248 | 49.4秒 |
| "A robot standing on a platform" | 18 | 750 | 1,184 | 17.5秒 |
| "An underwater coral reef scene" | 15 | 2,666 | 5,166 | 63.3秒 |
| "A simple mushroom on grass" | 9 | 8,237 | 16,224 | 6.6秒 |
| "火山地帯に宇宙船" | 22 | 10,466 | 20,618 | 20.5秒 |

手作りFew-shotサンプル（LLMシステムプロンプトで使用）:

| シーン | 説明 | ノード数 | 頂点数 | 三角形数 |
|-------|-------------|-------|----------|-----------|
| `sphere_on_ground` | 平面上の球体 (Union + Plane) | 4 | 1,270 | 2,448 |
| `snowman` | 3球体の雪だるま (SmoothUnion) | 8 | 422 | 840 |
| `castle_tower` | 胸壁付きの塔 (PolarRepeat) | 11 | 1,030 | 2,244 |
| `alien_mushroom_forest` | キノコグリッド (RepeatFinite + Torusステム) | 9 | 4,167 | 7,854 |
| `twisted_pillar` | ねじれた箱 + 浮遊する中空球 (Twist + Onion) | 7 | 510 | 968 |
| `mechanical_gear` | 歯と軸穴のあるギア (PolarRepeat + Subtraction) | 9 | 465 | 912 |

シーンJSONファイルは [`server/examples/scenes/`](server/examples/scenes/) に格納されています。

### クイックスタート（サーバー）

```bash
# 1. Pythonバインディングをビルド
cd /path/to/ALICE-SDF
python -m venv .venv && source .venv/bin/activate
maturin develop --features python

# 2. サーバー依存関係をインストール
pip install -r server/requirements.txt

# 3. APIキーを設定
export ANTHROPIC_API_KEY="sk-..."   # Claude用
export GOOGLE_API_KEY="AI..."       # Gemini用

# 4. サーバー起動
uvicorn server.main:app --reload

# 5. テキストから3D生成
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "雪だるま", "provider": "gemini", "resolution": 64}' \
  -o snowman.glb

# 6. ブラウザビューアを開く
open http://localhost:8000/api/viewer
```

### LLMプロバイダー

| プロバイダー | モデル | 速度 | 最適な用途 |
|----------|-------|-------|----------|
| Claude | Haiku 4.5 | ~2-5秒 | シンプルなシーン、高速イテレーション |
| Claude | Sonnet 4.5 | ~5-15秒 | 複雑なシーン、高精度 |
| Gemini | 2.5 Flash | ~5-50秒 | 複雑なシーン（思考モデル） |
| Gemini | 2.5 Pro | ~10-60秒 | 最高品質 |

### パフォーマンスバジェット

| ステップ | 時間 | 備考 |
|------|------|-------|
| LLM推論 | 2-60秒 | モデルと複雑さに依存 |
| JSON解析 | <1ms | serde_json |
| SDFコンパイル | ~1ms | SdfNode → CompiledSdf |
| メッシュ生成 (res=64) | ~45ms | 並列マーチングキューブ |
| GLBエクスポート | ~5ms | |
| **合計（LLM除く）** | **<55ms** | リアルタイム対応可能 |

### 堅牢性機能

- **JSON修復**: 切り詰められたLLM出力の括弧自動補完
- **構造バリデーション**: ブーリアン演算(a/b)とトランスフォーム(child)をRust serdeの前に事前検証
- **フィードバック付きリトライ**: エラーメッセージをLLMにフィードバックして最大2回リトライ
- **レート制限処理**: 429エラー時の自動待機リトライ
- **複雑度制約**: システムプロンプトでシーンを15-20ノード、ネスト深度≤6に制限

### サーバーディレクトリ構造

```
server/
├── main.py                  # FastAPIアプリ、REST + WebSocketエンドポイント
├── config.py                # APIキー、モデル設定（環境変数）
├── models.py                # Pydantic リクエスト/レスポンスモデル
├── services/
│   ├── llm_service.py       # Claude/Gemini API（リトライロジック付き）
│   └── sdf_service.py       # alice_sdfラッパー（パース、メッシュ、エクスポート）
├── prompts/
│   ├── system_prompt.py     # 36ノードタイプのSDF文法（LLM用）
│   └── examples.py          # 6つのFew-shotサンプル
├── examples/
│   └── scenes/              # ビルド済みシーンJSONファイル
│       ├── sphere_on_ground.json
│       ├── snowman.json
│       ├── castle_tower.json
│       ├── alien_mushroom_forest.json
│       ├── twisted_pillar.json
│       └── mechanical_gear.json
├── static/
│   └── viewer.html          # Three.js GLBビューア
├── tests/
│   ├── test_api.py          # 7つのAPIエンドポイントテスト
│   ├── test_llm_service.py  # 17のJSON抽出/バリデーションテスト
│   └── test_sdf_service.py  # 13のSDFパイプラインテスト
└── requirements.txt
```

### テスト実行

```bash
source .venv/bin/activate
python -m pytest server/tests/ -v   # 37テスト、全パス
```

## ALICE-View（リアルタイム3Dビューア）

**[ALICE-View](../ALICE-View)** はwgpuで構築されたネイティブGPUレイマーチングビューアです。WGSLトランスパイルにより、メッシュ変換なしでSDFツリーをGPU上で直接レンダリングします。

```
SDF JSON → ALICE-SDF (WGSLトランスパイル) → wgpu GPUレイマーチング → リアルタイム3D
              ~1ms                               60 FPS
```

### 機能

- **GPUレイマーチング** — SdfNodeツリーをWGSLシェーダーにトランスパイル、GPU上でピクセルごとに評価
- **ドラッグ&ドロップ** — `.json` / `.asdf` / `.asdf.json` ファイルをウィンドウにドロップ
- **ファイルダイアログ** — File > Open (Ctrl+O) フォーマットフィルター付き
- **カメラ操作** — マウスオービット、スクロールズーム、WASD移動
- **ライブSDFパネル** — ノード数、レイマーチングパラメータ（最大ステップ、イプシロン、AO）

### サポートフォーマット

| 拡張子 | フォーマット | 説明 |
|-----------|--------|-------------|
| `.json` | SDF JSON | Text-to-3Dパイプライン出力、Few-shotサンプル |
| `.asdf.json` | ALICE SDF JSON | ネイティブALICE-SDF JSONフォーマット |
| `.asdf` | ALICE SDFバイナリ | CRC32付きコンパクトバイナリ |
| `.alice` / `.alz` | ALICEレガシー | 手続き型コンテンツ（Perlin、Fractal） |

### クイックスタート

```bash
cd /path/to/ALICE-View

# 特定のファイルを開く
cargo run --bin alice-view -- path/to/scene.json

# 空で起動してファイルをドラッグ&ドロップ
cargo run --bin alice-view
```

### キーボードショートカット

| キー | アクション |
|-----|--------|
| `W/A/S/D` | カメラ移動 |
| `マウスドラッグ` | カメラオービット |
| `スクロール` | ズームイン/アウト |
| `Ctrl+O` | ファイルダイアログを開く |
| `Q` | 終了 |

### Text-to-3D結果の閲覧

Text-to-3Dパイプラインで生成されたシーンJSONファイルを直接閲覧できます:

```bash
# 生成シーンを表示
cargo run --bin alice-view -- /path/to/ALICE-SDF/server/examples/scenes/snowman.json

# または以下のファイルをウィンドウにドラッグ:
#   server/examples/scenes/castle_tower.json
#   server/examples/scenes/mechanical_gear.json
#   server/examples/scenes/alien_mushroom_forest.json
```

---

## コアコンセプト

### SDF（符号付き距離関数）

SDFは任意の点から表面までの最短距離を返します:
- **負** = 形状の内部
- **ゼロ** = 表面上
- **正** = 形状の外部

### SdfNodeツリー構造

```
SdfNode
  |-- プリミティブ (53): Sphere, Box3D, Cylinder, Torus, Plane, Capsule, Cone, Ellipsoid,
  |                    RoundedCone, Pyramid, Octahedron, HexPrism, Link, Triangle, Bezier,
  |                    RoundedBox, CappedCone, CappedTorus, InfiniteCylinder, RoundedCylinder,
  |                    TriangularPrism, CutSphere, CutHollowSphere, DeathStar, SolidAngle,
  |                    Rhombus, Horseshoe, Vesica, InfiniteCone, Heart, Gyroid,
  |                    Tube, Barrel, Diamond, ChamferedCube, SchwarzP, Superellipsoid, RoundedX,
  |                    Pie, Trapezoid, Parallelogram, Tunnel, UnevenCapsule, Egg,
  |                    ArcShape, Moon, CrossShape, BlobbyCross, ParabolaSegment,
  |                    RegularPolygon, StarPolygon, Stairs, Helix
  |-- 演算: Union, Intersection, Subtraction, SmoothUnion, SmoothIntersection, SmoothSubtraction
  |-- トランスフォーム (4): Translate, Rotate, Scale, ScaleNonUniform
  |-- モディファイア (15): Twist, Bend, RepeatInfinite, RepeatFinite, Noise, Round, Onion, Elongate,
  |                   Mirror, Revolution, Extrude, Taper, Displacement, PolarRepeat, Symmetry
  +-- WithMaterial: PBRマテリアル割り当て（距離評価に対して透過的）
```

## インストール

### Rust

```bash
cargo add alice-sdf
```

### Python

```bash
pip install alice-sdf
```

## 使い方

### Rust

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
```

## マテリアルシステム

glTF 2.0、UE5、Unity HDRP、Godot互換のPBRメタリック-ラフネスマテリアルシステム。

### マテリアルプロパティ

| プロパティ | 型 | 説明 |
|----------|------|-------------|
| `base_color` | `[f32; 4]` | RGBAベースカラー（リニア空間） |
| `metallic` | `f32` | 0.0 = 誘電体、1.0 = 金属 |
| `roughness` | `f32` | 0.0 = 鏡面、1.0 = 拡散 |
| `emission` | `[f32; 3]` | 発光色（RGB） |
| `emission_strength` | `f32` | 発光強度乗数 |
| `opacity` | `f32` | 0.0 = 透明、1.0 = 不透明 |
| `ior` | `f32` | 屈折率（ガラス=1.5、水=1.33） |
| `normal_scale` | `f32` | ノーマルマップ強度 |

### 使用例

```rust
use alice_sdf::prelude::*;

// マテリアルを作成
let gold = Material::metal("Gold", 1.0, 0.766, 0.336, 0.3);
let glass = Material::glass("Glass", 1.5);
let glow = Material::emissive("Neon", 0.0, 1.0, 0.0, 10.0);

// マテリアルライブラリ
let mut lib = MaterialLibrary::new();
let gold_id = lib.add(gold);

// マテリアルを形状に割り当て
let sphere = SdfNode::sphere(1.0).with_material(gold_id);

// AAA頂点フォーマットでメッシュ生成（UV、タンジェント、カラー、material_id）
let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &MarchingCubesConfig::aaa(64));
```

### 頂点フォーマット

AAA品質レンダリングに必要な全属性を含むメッシュ頂点:

| 属性 | 型 | 説明 |
|-----------|------|-------------|
| `position` | `Vec3` | 3D位置 |
| `normal` | `Vec3` | 表面法線 |
| `uv` | `Vec2` | トリプラナー投影テクスチャ座標 |
| `tangent` | `Vec4` | タンジェント（xyz=方向、w=利き手） |
| `color` | `[f32; 4]` | 頂点カラー（RGBAリニア） |
| `material_id` | `u32` | マテリアルライブラリインデックス |

## アニメーションシステム

リアルタイム変形、モーフィング、シネマティックシーケンス向けのキーフレームベースSDFパラメータアニメーション。

### 機能

- **補間モード**: Linear、Cubic Bezier（Hermite）、Step
- **ループモード**: Once、Loop、PingPong
- **タイムライン**: 名前付きパラメータによる複数トラック
- **AnimatedSdf**: 時間変化するSDF評価
- **モーフ**: 2つのSDF形状間のスムーズブレンド

### 使用例

```rust
use alice_sdf::prelude::*;

// バウンドする球体アニメーションを作成
let sphere = SdfNode::sphere(1.0);

let mut timeline = Timeline::new("bounce");

let mut ty = Track::new("translate.y").with_loop(LoopMode::PingPong);
ty.add_keyframe(Keyframe::new(0.0, 0.0));
ty.add_keyframe(Keyframe::cubic(0.5, 3.0, 0.0, 0.0));
ty.add_keyframe(Keyframe::new(1.0, 0.0));
timeline.add_track(ty);

let animated = AnimatedSdf::new(sphere, timeline);

// 時刻t=0.25で評価
let node_at_t = animated.evaluate_at(0.25);
let distance = eval(&node_at_t, Vec3::ZERO);

// 2つの形状間をモーフ
let sphere = SdfNode::sphere(1.0);
let cube = SdfNode::box3d(1.0, 1.0, 1.0);
let morphed = morph(&sphere, &cube, 0.5); // 50%ブレンド
```

### サポートされるトラック名

| トラック | 説明 |
|-------|-------------|
| `translate.x/y/z` | 平行移動オフセット |
| `rotate.x/y/z` | オイラー回転（ラジアン） |
| `scale` | 均一スケール係数 |
| `twist` | ねじり強度 |
| `bend` | 曲げ曲率 |

## ファイルフォーマット

### .asdf（ALICE SDFバイナリ）

CRC32整合性チェック付きコンパクトバイナリフォーマット。

```
ヘッダー（16バイト）:
  - マジック: "ASDF"（4バイト）
  - バージョン: u16（2バイト）
  - フラグ: u16（2バイト）
  - ノード数: u32（4バイト）
  - CRC32: u32（4バイト）

ボディ:
  - Bincodeシリアライズ済みSdfTree
```

### .asdf.json（ALICE SDF JSON）

デバッグ、相互運用性、LLM生成用の人間が読めるJSONフォーマット。

```json
{
  "version": "0.1.0",
  "root": {
    "Subtraction": {
      "a": {"Sphere": {"radius": 1.0}},
      "b": {"Box3d": {"half_extents": [1.5, 1.5, 1.5]}}
    }
  },
  "metadata": null
}
```

これはText-to-3Dパイプラインで使用されるのと同じフォーマットです — LLMがこのJSONを直接生成します。

### .obj（Wavefront OBJ）

マテリアル(.mtl)サポート付き標準メッシュフォーマット。

```rust
use alice_sdf::prelude::*;

let mesh = sdf_to_mesh(&shape, min, max, &MarchingCubesConfig::aaa(64));
export_obj(&mesh, "model.obj", &ObjConfig::default(), Some(&mat_lib))?;
```

### .glb（glTF 2.0バイナリ）

PBRマテリアル付き業界標準3Dフォーマット。UE5、Unity、Blender、Godot、Webビューアと互換。

```rust
use alice_sdf::prelude::*;

let mesh = sdf_to_mesh(&shape, min, max, &MarchingCubesConfig::aaa(64));
export_glb(&mesh, "model.glb", &GltfConfig::aaa(), Some(&mat_lib))?;
```

### アセットパイプラインまとめ

| フォーマット | インポート | エクスポート | マテリアル | 説明 |
|--------|--------|--------|-----------|-------------|
| `.asdf` | 対応 | 対応 | - | ネイティブSDFバイナリ（CRC32） |
| `.asdf.json` | 対応 | 対応 | - | ネイティブSDF JSON |
| `.obj` | 対応 | 対応 | .mtl | Wavefront OBJ（汎用DCCツール） |
| `.glb` | - | 対応 | PBR | glTF 2.0バイナリ（ゲームエンジン） |
| `.fbx` | - | 対応 | PBR | FBX 7.4 ASCII/バイナリ（DCCツール） |
| `.usda` | - | 対応 | UsdPreviewSurface | USD ASCII（Pixar/Omniverse/Houdini/Maya/Blender） |
| `.abc` | - | 対応 | - | Alembic Ogawaバイナリ（Maya/Houdini/Nuke/Blender） |
| `.nanite` | - | 対応 | - | UE5 Nanite階層クラスタバイナリ + JSONマニフェスト |

## アーキテクチャ

ALICE-SDFは13層アーキテクチャを使用しています。各SDF機能は全レイヤーにまたがって実装されています:

```
Layer 1:  types.rs          -- SdfNode列挙型（AST定義）
Layer 2:  primitives/       -- SDF数学公式（Inigo Quilez）
Layer 3:  eval/             -- 再帰インタプリタ
Layer 4:  compiled/opcode   -- VM用OpCode列挙型
Layer 5:  compiled/instr    -- 命令エンコーディング（32バイトアライン）
Layer 6:  compiled/compiler -- AST → 命令コンパイル
Layer 7:  compiled/eval     -- スタックベースVMエバリュエータ
Layer 8:  compiled/eval_simd-- SIMD 8-wideエバリュエータ（AVX2/NEON）
Layer 9:  compiled/eval_bvh -- BVH加速エバリュエータ（AABBプルーニング）
Layer 10: compiled/glsl     -- GLSLトランスパイラ（Unity/OpenGL/Vulkan）
Layer 11: compiled/wgsl     -- WGSLトランスパイラ（WebGPU）
Layer 12: compiled/hlsl     -- HLSLトランスパイラ（DirectX/Unreal）
Layer 13: compiled/jit      -- JITネイティブコード（Cranelift）
Layer 14: crispy.rs         -- ハードウェアネイティブ数学（ブランチレス、BitMask64、BloomFilter）
```

### 評価モード

| モード | 説明 | ユースケース |
|------|-------------|----------|
| **インタプリタ** | 再帰ツリーウォーク | デバッグ、プロトタイピング |
| **コンパイルVM** | スタックベースバイトコード | 汎用 |
| **SIMD 8-wide** | 8点並列処理（Vec3x8） | バッチ評価 |
| **BVH加速** | AABB空間プルーニング | 複雑なシーン |
| **SoAバッチ** | Structure-of-Arraysメモリレイアウト | キャッシュ最適SIMDバッチ |
| **JITネイティブ** | Craneliftマシンコード | 最大スループット |
| **GPUコンピュート** | WGSLコンピュートシェーダー | 大量バッチ |

### シェーダートランスパイラ

| ターゲット | 出力 | ユースケース |
|--------|--------|----------|
| **GLSL** | OpenGL/Vulkanシェーダー | Unity、カスタムエンジン |
| **WGSL** | WebGPUシェーダー | ブラウザ、wgpu |
| **HLSL** | DirectXシェーダー | Unreal Engine、DirectX |

#### エンジン固有シェーダーエクスポート

| メソッド | 出力 | ターゲットエンジン |
|--------|--------|---------------|
| `HlslShader::export_ue5_material_function()` | `.ush` Material Functionインクルード | Unreal Engine 5（Custom Expression） |
| `GlslShader::export_unity_shader_graph()` | `.hlsl` Custom Functionノード | Unity Shader Graph（HDRP/URP） |

## メッシュモジュール

### 変換

| 関数 | 説明 |
|----------|-------------|
| `sdf_to_mesh()` | 並列マーチングキューブによるSDF→メッシュ（Zスラブ並列化） |
| `sdf_to_mesh_compiled()` | コンパイルVMパス — SIMDバッチグリッド評価 + グリッド有限差分法線 |
| `marching_cubes_compiled()` | `eval_compiled_batch_simd_parallel`グリッド評価のコンパイルMC |
| `adaptive_marching_cubes()` | オクツリー適応型MC（インタプリタ）— 三角形60-80%削減 |
| `adaptive_marching_cubes_compiled()` | オクツリー適応型MC（コンパイルVM）— 2-5倍高速 |
| `mesh_to_sdf()` | カプセル近似によるメッシュ→SDF（エッジベース） |
| `mesh_to_sdf_exact()` | BVH精密距離によるメッシュ→SDF（O(log n)クエリ） |

### 高度な機能

| 機能 | モジュール | 説明 |
|---------|--------|-------------|
| **エルミートデータ** | `mesh/hermite` | Dual Contouring用の位置+法線抽出 |
| **プリミティブフィッティング** | `mesh/primitive_fitting` | メッシュデータ内の球/箱/円柱検出（CSG再構成用） |
| **Naniteクラスタ** | `mesh/nanite` | UE5 Nanite互換階層クラスタ生成 |
| **LOD生成** | `mesh/lod` | 効率的レンダリング用LODチェーン生成 |
| **デシメーションLOD** | `mesh/lod` | 高解像度ベースメッシュからのプログレッシブデシメーションLOD |
| **適応型MC** | `mesh/sdf_to_mesh` | 表面適応細分化を備えたオクツリーベースマーチングキューブ |
| **メッシュデシメーション** | `mesh/decimate` | UV/タンジェント/マテリアル境界保護付きQEMデシメーション |
| **凸分解** | `mesh/collision` | 物理用V-HACD体積凸分解 |
| **コリジョンプリミティブ** | `mesh/collision` | AABB、バウンディングスフィア、凸包、簡略化コリジョン |
| **ライトマップUV** | `mesh/lightmap` | 自動ライトマップUV生成（UVチャンネル1） |
| **頂点最適化** | `mesh/optimize` | 頂点キャッシュ最適化と重複排除 |
| **メッシュBVH** | `mesh/bvh` | 精密符号付き距離クエリ用バウンディングボリューム階層 |
| **マニフォールドバリデーション** | `mesh/manifold` | トポロジーバリデーション、修復、品質メトリクス |
| **UV展開** | `mesh/uv_unwrap` | LCSMコンフォーマルUV展開（シーム検出、チャートパッキング） |

### マニフォールドメッシュ保証

物理、レンダリング、3Dプリントに適した水密マニフォールドメッシュを保証。

```rust
use alice_sdf::prelude::*;

let mesh = sdf_to_mesh(&shape, min, max, &MarchingCubesConfig::default());

// バリデーション
let report = validate_mesh(&mesh);
println!("{}", report); // 完全なバリデーションレポートを表示

// 修復
let repaired = MeshRepair::repair_all(&mesh, 1e-6);

// 品質メトリクス
let quality = compute_quality(&repaired);
println!("平均アスペクト比: {}", quality.avg_aspect_ratio);
```

| 関数 | 説明 |
|----------|-------------|
| `validate_mesh()` | 非マニフォールドエッジ検出、境界エッジ、退化三角形、重複頂点、法線整合性 |
| `MeshRepair::remove_degenerate_triangles()` | ゼロ面積三角形の除去 |
| `MeshRepair::merge_duplicate_vertices()` | 空間ハッシュベースの頂点溶接 |
| `MeshRepair::fix_normals()` | 不整合な巻き順の修正 |
| `MeshRepair::repair_all()` | 全修復を順次実行 |
| `compute_quality()` | アスペクト比と面積の統計 |

## テクスチャフィッティング（テクスチャ→数式変換）

ビットマップテクスチャ（PNG/JPG）を解像度非依存の手続き的ノイズ公式に変換します。フィッティング済みの公式は、元の画像なしで**任意の解像度**でGPU上でレンダリングできます。

```
texture(u,v) ≈ bias + Σᵢ aᵢ · noise(uv · fᵢ + φᵢ, seedᵢ)
```

CPUノイズ実装（`hash_noise_3d`）はWGSL/HLSL/GLSLのGPU版と完全に一致し、CPUフィッティング=GPUレンダリングを保証します。

### パイプライン

1. 画像読み込み → グレースケールf32
2. DCT周波数分析 → 支配的帯域
3. 貪欲オクターブごとのフィッティング（Nelder-Mead、SIMD f32x8 + rayon並列）
4. JSONパラメータおよび/またはスタンドアロンシェーダー関数としてエクスポート

### CLI

```bash
# 基本: テクスチャフィッティングして結果を表示
alice-sdf texture-fit granite.png

# JSONパラメータ + HLSLシェーダーをエクスポート
alice-sdf texture-fit granite.png -o params.json --shader hlsl --shader-output granite.hlsl

# 高品質: より多いオクターブ、より高いPSNR目標
alice-sdf texture-fit marble.png --octaves 12 --target-psnr 35.0 --iterations 2000
```

### Rust API

```rust
use alice_sdf::texture::{fit_texture, generate_shader, ShaderLanguage, TextureFitConfig};
use std::path::Path;

let config = TextureFitConfig {
    max_octaves: 8,
    target_psnr_db: 28.0,
    iterations_per_octave: 500,
    tileable: true,
};

let result = fit_texture(Path::new("granite.png"), &config).unwrap();
println!("PSNR: {:.1} dB, {} オクターブ", result.psnr_db, result.octaves[0].len());

// スタンドアロンWGSLシェーダーを生成
let shader = generate_shader(&result, ShaderLanguage::Wgsl, "granite.png");
```

### 出力シェーダー言語

| ターゲット | 関数シグネチャ | ユースケース |
|--------|-------------------|----------|
| **WGSL** | `fn procedural_texture(uv: vec2<f32>) -> f32` | WebGPU、wgpu |
| **HLSL** | `float procedural_texture(float2 uv)` | Unity、Unreal、DirectX |
| **GLSL** | `float procedural_texture(vec2 uv)` | OpenGL、Vulkan |

## レイマーチング

専用最適化を備えたSDF-レイ交差のスフィアトレーシング:

| 関数 | 説明 |
|----------|-------------|
| `raymarch()` | スフィアトレーシングによる単一レイ交差 |
| `raymarch_batch()` | バッチレイ評価 |
| `raymarch_batch_parallel()` | Rayonによる並列バッチ |
| `render_depth()` | 深度バッファレンダリング |
| `render_normals()` | 法線マップレンダリング |

機能: 専用Shadow/AOループ（法線計算をスキップ）、ハードシャドウ用早期終了、`RaymarchConfig`による設定可能な反復回数制限。

## FFI & 言語バインディング

### C/C++（`include/alice_sdf.h`）

```c
#include "alice_sdf.h"

AliceSdfHandle sdf = alice_sdf_sphere(1.0);
float dist = alice_sdf_eval(sdf, 0.5, 0.0, 0.0);
```

### C# / Unity（`bindings/AliceSdf.cs`）

```csharp
using AliceSdf;

var sdf = AliceSdf.Sphere(1.0f);
float dist = sdf.Eval(new Vector3(0.5f, 0f, 0f));
```

### Python（PyO3）

```bash
pip install alice-sdf  # または: maturin develop --features python
```

### FFIパフォーマンス階層

| 関数 | 速度 | ユースケース |
|----------|-------|----------|
| `alice_sdf_eval_soa` | 最速 | 物理、パーティクル、トレーシング |
| `alice_sdf_eval_compiled_batch` | 高速 | 汎用バッチ評価 |
| `alice_sdf_eval_batch` | 中速 | 利便性（自動コンパイル） |
| `alice_sdf_eval` | 低速 | デバッグ専用 |

## フィーチャーフラグ

| フィーチャー | 説明 | 依存関係 |
|---------|-------------|--------------|
| `cli`（デフォルト） | コマンドラインインターフェース | clap |
| `python` | Pythonバインディング | pyo3, numpy |
| `jit` | JITネイティブコードコンパイル | cranelift |
| `gpu` | WebGPUコンピュートシェーダー | wgpu, pollster, bytemuck |
| `glsl` | GLSLシェーダートランスパイラ | - |
| `hlsl` | HLSLシェーダートランスパイラ | - |
| `ffi` | C/C++/C# FFIバインディング | lazy_static |
| `unity` | Unity統合 | ffi + glsl |
| `unreal` | Unreal Engine統合 | ffi + hlsl |
| `all-shaders` | 全シェーダートランスパイラ | gpu + hlsl + glsl |
| `texture-fit` | テクスチャ→数式変換 | image, rayon, wide |

```bash
# 例
cargo build --features "jit,gpu"          # JIT + GPU
cargo build --features unity              # Unity (FFI + GLSL)
cargo build --features unreal             # Unreal (FFI + HLSL)
cargo build --features "all-shaders,jit"  # 全部入り
```

## テスト

全モジュールにまたがる574以上のテスト（プリミティブ、演算、トランスフォーム、モディファイア、コンパイラ、エバリュエータ、BVH、I/O、メッシュ、シェーダートランスパイラ、マテリアル、アニメーション、マニフォールド、OBJ、glTF、FBX、USD、Alembic、Nanite、UV展開、コリジョン、デシメーション、LOD、適応型MC、crispyユーティリティ、BloomFilter）。`--features jit`で590以上のテスト（JITスカラーおよびJIT SIMDバックエンド含む）。

```bash
cargo test
```

## パフォーマンス

Apple M3 Max、Rust 1.75+、`--release`ビルドでベンチマーク。

### 単一点評価

| プリミティブ | 時間 |
|-----------|------|
| Sphere | 6.1 ns |
| Box3D | 5.0 ns |
| Cylinder | 8.0 ns |
| Torus | 9.3 ns |

| 演算 | 時間 |
|-----------|------|
| Union（2ノード） | 13.3 ns |
| Smooth Union | 21.4 ns |
| 複合ツリー（5ノード） | 12.6 ns |
| 複合ツリー（10ノード） | 51.5 ns |
| 複合ツリー（20ノード） | 66.5 ns |

### バッチ評価比較（100万点）

| モード | スループット | ns/点 | フィーチャー |
|------|------------|----------|---------|
| CPU JIT SIMD | 977 M/s | 1.0 ns | `--features jit` |
| CPUスカラー | 307 M/s | 3.3 ns | デフォルト |
| CPU SIMD (VM) | 252 M/s | 4.0 ns | デフォルト |
| GPUコンピュート | 101 M/s | 9.9 ns | `--features gpu` |

### JITコンパイル

JITコンパイラはCraneliftを使用してネイティブSIMDマシンコードを生成し、最高スループットを実現します。オリジナルの15プリミティブはJITスカラーとJIT SIMD（8-wide）バックエンドの両方で完全サポートされています。16の新しいIQプリミティブは、インタプリタ、コンパイルVM、シェーダートランスパイラモードでサポートされています。

**Deep Fried v2最適化:**
- **Division Exorcism** - 全ランタイム除算をコンパイル時に逆数乗算として事前計算
- **ブランチレスSIMDセレクション** - `bitcast`/`sshr`/`bitselect`による符号ビット抽出（SSE/AVX/NEONでゼロオーバーヘッド）
- **FMAフュージョン** - 複雑なプリミティブ（Cone、RoundedCone、Pyramid）でのレイテンシ削減のための融合積和演算

### crispy.rs — ハードウェアネイティブ数学ユーティリティ

ホットインナーループ向けの低レベルブランチレス演算。スループットのためにsub-ULP精度を犠牲にします。

| 関数 | 説明 |
|----------|-------------|
| `fast_recip(x)` | ハードウェアrcpss + Newton-Raphsonによる高速`1/x`（~0.02%誤差） |
| `fast_inv_sqrt(x)` | Quake III逆平方根 + NRイテレーション（~0.175%誤差） |
| `fast_normalize_2d(gx, gz)` | 高速逆平方根を使用した2D勾配正規化 |
| `select_f32(cond, a, b)` | ビット操作によるブランチレスcmov |
| `branchless_min/max/clamp/abs` | `select_f32`によるゼロブランチ算術 |
| `BitMask64` | 64要素バッチマスク（AND/OR/NOT/popcntをハードウェアで） |
| `BloomFilter` | 4KB Bloom Filter、FNV-1aダブルハッシュ、O(1)メンバーシップテスト |
| `fnv1a_hash(data)` | FNV-1a 64ビットハッシュ（高速、良分布） |

### コンパイルマーチングキューブのパフォーマンス

コンパイルMCパス（`sdf_to_mesh_compiled`）は2つの主要な最適化を適用:

1. **SIMDバッチグリッド評価** — 全グリッドポイントを`Vec<Vec3>`に収集し、`eval_compiled_batch_simd_parallel`（8-wide SIMD + Rayon）で一括評価。ポイントごとの関数呼び出しオーバーヘッドを排除。

2. **グリッド有限差分法線** — 頂点法線を隣接グリッド値（`values[x+1] - values[x-1]`）から計算。頂点ごとに6回の追加`eval_compiled`呼び出しの代わり。内部セルは法線計算にゼロeval呼び出し、境界セルは解析的法線にフォールバック。

```bash
# CLIベンチマーク実行
cargo run --features "cli,jit" --release -- bench --points 1000000
```

| 点数 | JIT SIMD | SIMD (VM) | 高速化率 |
|--------|----------|-----------|---------|
| 100K | 330 M/s | 197 M/s | 1.7x |
| 1M | 977 M/s | 252 M/s | 3.9x |

### SIMD 8-wide評価

| モード | 時間（8点） | 高速化率 |
|------|-----------------|---------|
| スカラー | 563 ns | 1.0x |
| SIMD | 143 ns | 3.9x |

### マーチングキューブ（Sphere、bounds +/-2.0）

| 解像度 | 時間 |
|------------|------|
| 16^3 | 140 us |
| 32^3 | 390 us |
| 64^3 | 1.64 ms |

### レイマーチング

| 形状 | レイあたりの時間 |
|-------|--------------|
| Sphere | 62 ns |
| 複合（smooth union + twist） | 178 ns |

### GPUコンピュート（WebGPU）

GPUモジュールは、繰り返し評価用の永続バッファプーリングを備えたWebGPUコンピュートシェーダーを提供します。

**GPU vs CPUの使い分け:**

| バッチサイズ | 推奨 | 理由 |
|------------|-------------|--------|
| < 100K | CPU JIT SIMD | GPU転送オーバーヘッドが支配的 |
| 100K - 1M | CPU JIT SIMD | M3 MaxではまだJITが高速 |
| > 1M | 両方テスト | ハードウェアと形状の複雑さに依存 |

注: GPUパフォーマンスはハードウェアにより大きく異なります。ディスクリートGPUではクロスオーバーポイントが低くなる場合があります。

```rust
use alice_sdf::prelude::*;
use alice_sdf::compiled::{GpuEvaluator, WgslShader, GpuBufferPool};

let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);

// GPUエバリュエータを作成（SDFをWGSLにコンパイル）
let gpu = GpuEvaluator::new(&shape).unwrap();

// ワンショットバッチ評価
let points: Vec<Vec3> = (0..100000)
    .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
    .collect();
let distances = gpu.eval_batch(&points).unwrap();

// 繰り返し評価用永続バッファプール（2-5倍高速）
let mut pool = gpu.create_buffer_pool(100000);
for frame in 0..60 {
    let distances = gpu.eval_batch_pooled(&points, &mut pool).unwrap();
}

// 自動チューニングバッチ（大バッチを最適チャンクに分割）
let distances = gpu.eval_batch_auto(&points, &mut pool).unwrap();
```

有効化: `cargo build --features gpu`

| モード | スループット（100万点） |
|------|---------------------|
| CPU JIT SIMD | ~977 M/s |
| CPU SIMD (VM) | ~252 M/s |
| GPUコンピュート | ~101 M/s |

## WebAssembly（ブラウザ）

ALICE-SDFはWebAssemblyでブラウザ上で動作し、WebGPU/Canvas2Dをサポートします。

### npmパッケージ（`@alice-sdf/wasm`）

```bash
npm install @alice-sdf/wasm
```

TypeScript型定義を完備。53プリミティブ、CSG演算、トランスフォーム、メッシュ変換、シェーダー生成（WGSL/GLSL）を全サポート。

### WASMデモのビルド

```bash
cd examples/wasm-demo
wasm-pack build --target web
python3 -m http.server 8080
# http://localhost:8080 を開く
```

### 機能

- **WebGPUコンピュート**: ハードウェアアクセラレーション評価（Chrome 113+、Edge 113+）
- **Canvas2Dフォールバック**: 旧ブラウザ向けCPUレイマーチング
- **リアルタイムビジュアライゼーション**: インタラクティブな形状編集とレンダリング

### ブラウザ互換性

| ブラウザ | WebGPU | Canvas2D |
|---------|--------|----------|
| Chrome 113+ | 対応 | 対応 |
| Edge 113+ | 対応 | 対応 |
| Firefox Nightly | 対応（フラグ） | 対応 |
| Safari 18+ | 対応 | 対応 |
| 旧ブラウザ | 非対応 | 対応 |

## ベンチマーク

評価モードを比較するベンチマークの実行:

```bash
# CPUベンチマーク（インタプリタ、SIMD、BVH）
cargo bench --bench sdf_eval

# JIT + SoAスループットベンチマーク
cargo bench --bench sdf_eval --features jit -- soa_throughput

# GPU vs CPU比較
cargo bench --bench gpu_vs_cpu --features "jit,gpu"

# CLIクイックベンチマーク
cargo run --features "cli,jit,gpu" --release -- bench --points 1000000

# HTMLレポートを表示
open target/criterion/report/index.html
```

## Unityデモ: SDF Universe

`unity-sdf-universe/` ディレクトリにはALICE-SDFの機能を紹介するフルUnityデモが含まれています:

**"5MBプロシージャルユニバース"** - わずか5MBのコードで完全なプロシージャルユニバースを実現。

### 機能

| 機能 | 説明 |
|---------|-------------|
| **1000万以上のパーティクル** | GPUコンピュートシェーダーパーティクルシステム（60+ FPS） |
| **4つのシーンタイプ** | Cosmic、Terrain、Abstract、Fractal |
| **無限解像度** | レイマーチング + プロシージャルテクスチャリング |
| **フラクタルダイブ** | x10,000以上のズームを持つ顕微鏡デモ |

### クイックスタート

```bash
# 1. Rustライブラリをビルド
cargo build --release

# 2. Unityにコピー
cp target/release/libalice_sdf.dylib unity-sdf-universe/Assets/Plugins/  # macOS

# 3. Unity 2022.3+で開く
# 4. Assets/Scenes/SdfUniverse.unityを開く
# 5. Playを押す
```

### フラクタルダイブ（顕微鏡デモ）

レイマーチングによるTRUE無限解像度のデモンストレーション:

- **SDF公式**: `Subtract(Box, Repeat(Cross))` - 単一の数学オブジェクト
- **レイマーチング**: ピクセルごとのSDF評価（128ステップ）
- **プロシージャルテクスチャリング**: FBMノイズからの色（ピクセル化なし）
- **[R]キー**: レイマーチングとパーティクルモードの切り替え

詳細は`unity-sdf-universe/README.md`を参照。

## VRChat統合

`vrchat-package/` ディレクトリはVRChat SDK向けのSDF世界・アバターパッケージを提供します。

- **ALICE-Shader** - 動的LOD付きHLSLレイマーチングカーネル
- **ALICE-Udon** - 純粋C#数学のUdonSharp SDFコライダー
- **ALICE-Baker v0.3** - `.asdf.json`から最適化シェーダー + Udonを生成するエディタツール
- **7つのサンプルワールド** - Basic、Cosmic、Fractal、Mix、DeformableWall、Mochi、TerrainSculpt

詳細は`vrchat-package/README.md`を参照。

## Unreal Engine 5統合

ALICE-SDFはHLSLトランスパイラとC FFIバインディングによるフルUE5サポートを提供します。

```bash
# プラグインDLLをビルド
cargo build --release --features unreal
```

- **HLSLトランスパイラ** - Custom Material Expressionノードの生成
- **C++ FFI** - `alice_sdf.h`ヘッダー付きネイティブプラグイン
- **Blueprintレディ** - ビジュアルスクリプティング用UFunctionラッパー

詳細なセットアップ手順は`docs/UNREAL_ENGINE.md`を参照。

## Godot統合

ALICE-SDFはglTF 2.0インポートとGDExtension FFI経由でGodotと連携します。

- **glTFパイプライン** - `.glb`をエクスポートしてGodotに直接インポート
- **GDNative/GDExtension** - C FFI経由で`libalice_sdf`をリンク
- **ビジュアルシェーダー** - GLSLトランスパイラ出力をシェーダーノードで使用

統合ガイドは`docs/GODOT_GUIDE.md`を参照。

## ドキュメント

| ドキュメント | 説明 |
|----------|-------------|
| [ALICE-View](../ALICE-View) | リアルタイムGPUレイマーチングビューア（wgpu、ドラッグ&ドロップ） |
| [クイックスタート](docs/QUICKSTART.md) | 全プラットフォーム向け5分入門ガイド |
| [アーキテクチャ](docs/ARCHITECTURE.md) | 13層アーキテクチャ詳細 |
| [APIリファレンス](docs/API_REFERENCE.md) | 完全なAPIリファレンス |
| [Unreal Engine](docs/UNREAL_ENGINE.md) | UE5セットアップ・統合ガイド |
| [Pythonガイド](docs/PYTHON_GUIDE.md) | Python・Blender統合 |
| [WASMガイド](docs/WASM_GUIDE.md) | WebAssemblyデプロイガイド |
| [Godotガイド](docs/GODOT_GUIDE.md) | Godot統合ガイド |
| [Unityセットアップ](unity-sdf-universe/SETUP_GUIDE.md) | Unityプロジェクトセットアップ |
| [VRChatパッケージ](vrchat-package/README.md) | VRChat SDK統合 |

## ライセンス

**オープンコアモデル** - クリエイターは無料、インフラは有償。

| コンポーネント | ライセンス | 用途 |
|-----------|---------|----------|
| **コアエンジン**（Rust） | MITライセンス | 自由に改変可能！ |
| **Unity統合** | ALICEコミュニティライセンス | インディー・ゲーム開発は無料 |
| **エンタープライズ / クラウドインフラ** | 商用ライセンス | 価格はお問い合わせください |

### 無料利用（ライセンス不要）

- 個人プロジェクト
- インディーゲーム開発（収益に関係なく）
- AAAゲームスタジオ（出荷ゲーム）
- 教育・研究
- オープンソースプロジェクト

### 商用ライセンスが必要

- メタバースプラットフォーム（10,000+ MAU）
- クラウドストリーミングサービス（SaaS/PaaS）
- インフラプロバイダー
- 競合製品

詳細は[LICENSE](LICENSE)（MIT）および[LICENSE-COMMUNITY](LICENSE-COMMUNITY)を参照。

**あなたが作成するコンテンツ（.asdfファイル、ワールド、ゲーム）は100%あなたのものです。ロイヤリティはありません。**

---

Copyright (c) 2025 Moroya Sakamoto
