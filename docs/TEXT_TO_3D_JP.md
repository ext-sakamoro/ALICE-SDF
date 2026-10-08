# Text-to-3D サーバーとビューア

[`server/`](../server/) の Python サーバーは LLM を通してテキストを SDF tree に変換し、
ALICE-View が結果を表示する どちらも crate の上に作ったアプリケーションで、`alice-sdf` ライブラリの一部ではない

[English](TEXT_TO_3D.md)

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

シーンJSONファイルは [`server/examples/scenes/`](../server/examples/scenes/) に格納されています。

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

**[ALICE-View](https://github.com/ext-sakamoro/ALICE-View)** はwgpuで構築されたネイティブGPUレイマーチングビューアです。WGSLトランスパイルにより、メッシュ変換なしでSDFツリーをGPU上で直接レンダリングします。

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

## LLM × 3D制作パイプライン（SDF + LOL + View + Physics）

4つのALICEプロジェクトを組み合わせることで、自然言語から物理シミュレーション付き3Dシーンまでの**エンドツーエンド**ワークフローが完成します:

```
ユーザー: 「シルクハットをかぶった雪だるま」
         │
         ▼
┌──────────────────┐  LOL DSL or JSON  ┌───────────────────┐  WGSL / GLB   ┌──────────────┐
│  LLM             │ ────────────────▶ │  ALICE-SDF        │ ────────────▶ │  ALICE-View  │
│  (Claude/Gemini) │                   │  parse → compile  │               │  GPUプレビュー│
│                  │                   │  → mesh / shader  │               │  60 FPS      │
└──────────────────┘                   └────────┬──────────┘               └──────────────┘
                                                │
                                                │ SdfField トレイト
                                                │ (feature = "physics")
                                                ▼
                                       ┌───────────────────┐
                                       │  ALICE-Physics     │
                                       │  Fix128 XPBD       │
                                       │  SDF CCD / 力場    │
                                       │  破壊 / 流体       │
                                       └───────────────────┘
```

| コンポーネント | 役割 |
|--------------|------|
| **[ALICE-LOL](https://github.com/ext-sakamoro/ALICE-LOL)** | LLM向けDSL — JSONより少ないトークンで低ハルシネーション率。`runtime_parser::parse_lol()` でLLMテキスト出力を `SdfNode` にランタイム変換 |
| **ALICE-SDF** | コアエンジン — SIMD/BVH/JIT評価、メッシュ生成（Marching Cubes / Dual Contouring）、GLSL/WGSL/HLSLトランスパイル、GLB/OBJ/STLエクスポート |
| **[ALICE-View](https://github.com/ext-sakamoro/ALICE-View)** | リアルタイムGPUレイマーチングビューア — JSON/ASDFファイルをドラッグ&ドロップで即座にプレビュー |
| **[ALICE-Physics](https://github.com/ext-sakamoro/ALICE-Physics)** | 決定論的128bit固定小数点物理エンジン — `SdfField` トレイトでSDF形状がそのまま衝突ジオメトリに。SDF CCD、力場、破壊、布、流体シミュレーション |

LLMで生成した形状は見た目だけではなく、**物理シミュレーション対応**です。`CompiledSdfField` ラッパーがSDFをO(1)衝突クエリ面として公開するため、凸分解なしで剛体・破壊・流体のインタラクションが可能です。

> **ブリッジ注意** — `alice-sdf = { version = "5", features = ["physics"] }` で alice-physics が crates.io から解決されます (1.12.0 以降) 詳細は [インストール](../README_JP.md#インストール) 参照

### クイックスタート

```bash
# 1. Text-to-3Dサーバー起動（LLMでLOL/JSON生成）
cd ALICE-SDF/server
python main.py

# 2. プロンプトをPOST — SDF JSONが返る
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "シルクハットをかぶった雪だるま", "format": "json"}'

# 3. リアルタイムで結果を確認
cd ALICE-View
cargo run --bin alice-view -- ../ALICE-SDF/server/output/latest.json
```

### プログラマティック（Rust）

```rust
use alice_lol::runtime_parser::parse_lol;
use alice_sdf::prelude::*;
use alice_sdf::physics_bridge::CompiledSdfField;

// LLM出力（テキスト） → SdfNode
let lol_text = r#"smooth_union(0.3, sphere(1.0), translate(0.0, 1.5, 0.0, sphere(0.7)))"#;
let scene = parse_lol(lol_text).unwrap();

// レンダリング用GPUシェーダー
let wgsl = alice_lol::to_wgsl(&scene);

// メッシュエクスポート
let mesh = alice_sdf::mesh::sdf_to_mesh(
    &scene,
    glam::Vec3::splat(-3.0),
    glam::Vec3::splat(3.0),
    &MeshConfig::default(),
);

// 物理対応の衝突形状（凸分解不要）
let field = CompiledSdfField::new(scene);
// field.distance(x, y, z)            → f32        (1回評価)
// field.distance_and_normal(x, y, z) → (f32, Vec3) (4回評価、四面体法)
```

### なぜJSONよりLOLか？

| 指標 | JSON (SdfNode) | LOL DSL |
|------|---------------|---------|
| 形状あたりトークン数 | ~120 | ~30 |
| LLMエラー率 | 高（括弧ネスト） | 低（関数呼び出しスタイル） |
| ランタイムパース | `serde_json` | `runtime_parser::parse_lol()` |
| コンパイル時マクロ | — | `lol! { ... }` |

複雑なシーンではLOLは**3〜4倍少ないトークン**で記述でき、LLMのコストとハルシネーションの両方を削減します。

---

