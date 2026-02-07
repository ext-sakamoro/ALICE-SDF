# ALICE-SDF for VRChat

**「ポリゴンよ、さらば。」**

ALICE-SDFは、**数式（SDF）で定義された無限に滑らかな曲面** をVRChatの世界に持ち込むためのオールインワン・パッケージです。
単に見えるだけでなく、**プレイヤーがその上に立ち、衝突し、インタラクトする** ことが可能です。

[English / 英語版](README.md)

---

## なぜ ALICE-SDF なのか？

従来の「3Dメッシュ」によるワールド制作には限界がありました。ALICE-SDFは、**数式** を送ることで世界を描画します。

1. **無限の解像度**: どれだけカメラを近づけても、ポリゴンの角（カクつき）は一切見えません。完全な球体、滑らかな結合が可能です。
2. **圧倒的なデータ圧縮**: 複雑な有機的形状も、数式ならわずか数キロバイト。ワールド容量を劇的に削減します。
3. **リアルタイム変形**: シェーダーパラメータを変えるだけで、形状がグニャグニャとモーフィングします。メッシュの頂点移動では不可能な表現です。
4. **物理判定 (Collider) 完全対応**: 「レイマーチングは当たり判定がない」という常識を覆しました。UdonSharpにより、数式の床の上に立つことができます。

---

## インストール

### Unity Package Manager 経由 (推奨)

1. Unityで **Window > Package Manager** を開く
2. **+** ボタン > **Add package from disk...** をクリック
3. ALICE-SDFフォルダ内の `package.json` を選択

または `Packages/manifest.json` に git URL を直接追加:

```json
{
  "dependencies": {
    "com.alice.sdf": "https://github.com/sakamoro/ALICE-SDF.git?path=vrchat-package"
  }
}
```

### .unitypackage 経由 (レガシー)

1. `ALICE-SDF-VRChat.unitypackage` をダウンロード
2. Unityで **Assets > Import Package > Custom Package...** を選択
3. 全てを選択してインポート

### 動作環境

- Unity 2022.3.x (VRChat 推奨バージョン)
- VRChat SDK3 - Worlds
- **UdonSharp** (必須: 物理判定のため)

---

## ALICE-Baker v0.3 (Deep Fried Edition)

**「数式を書くのが難しい？」心配無用です。**

付属の **ALICE-Baker** ツールが、JSON定義ファイルから「最適化されたシェーダー」と「Udonスクリプト」を全自動生成します。

1. Unityメニューから **Window > ALICE-SDF Baker** を開く
2. `.asdf.json` をドラッグ＆ドロップ（またはJSONテキストをペースト）
3. **Bake!** ボタンをクリック

これだけで、**Prefab** が生成されます。あとはシーンに置くだけです。

### "Deep Fried" 最適化とは？

VRChatのUdon VMは、通常のC#に比べて実行速度に制約があります。ALICE-Baker v0.3は、生成されるコードに対して**狂気的な最適化（Deep Frying）**を施します。

| 最適化手法 | 効果 (対 通常実装比) |
|------------|----------------------|
| **命令融合 (Instruction Fusion)** | 計算式をインライン展開し、一時変数のメモリ確保を削減。 |
| **除算の悪魔払い (Division Exorcism)** | 重い「割り算」命令を排除。逆数（`1/k`）を事前計算し「掛け算」に変換。 |
| **Udonスカラ展開** | `Vector3` の構造体コピーはUdonでは高コストです。これを `float x,y,z` のスカラ演算に分解し、VMオーバーヘッドを回避。 |
| **関数コール削除** | `Sdf.Union()` などの関数呼び出しを `Mathf.Min()` へ直書き展開。メソッド呼び出しコストをゼロに。 |
| **Smooth演算の完全展開** | `SmoothUnion/Intersection/Subtraction` を数式レベルでインライン化。`inv_k` をコンパイル時定数として埋め込み。 |

これにより、**50HzのFixedUpdate内でも余裕で動作する** 高速な物理判定を実現しました。

---

## 3つのコアコンポーネント

### 1. ALICE-Shader (描画エンジン)

ピクセル単位で距離関数をレイマーチング描画します。

- **SV_Depth 対応**: VRMアバターや他のオブジェクトとの前後関係（オクルージョン）も完璧に処理します。「アバターが壁に埋まると描画がおかしくなる」というレイマーチング特有の問題を解消。
- **Deep Fried LOD**: カメラ距離に応じて、レイのステップ数と精度を動的に調整します。遠くのオブジェクトは負荷が激減します。

| 距離 | ステップ | Epsilon | 品質 |
|------|---------|---------|------|
| < 20m | 128 | 0.0001 | High |
| 20–60m | 64 | 0.001 | Medium |
| > 60m | 32 | 0.005 | Low |

### 2. ALICE-Udon (物理エンジン)

シェーダーと同じ数式をC#で計算し、プレイヤーを押し出します。

- プレイヤーが数式の内部（`d < 0`）に侵入すると、法線ベクトル（`∇SDF`）を計算し、表面まで押し戻します。
- 1回のティックにつき SDF評価 1回 + 勾配サンプル 6回 = **計7回の評価のみ**。典型的なコストは **< 0.01ms/プレイヤー**。
- ネイティブプラグイン（DllImport）を使用しないため、**Quest単体でも動作可能なロジック** です。

### 3. ALICE-Math (共通ライブラリ)

HLSLとUdonSharpの両方で「全く同じ挙動」をするように設計された数学ライブラリ群。

- `SmoothUnion` や `Twist` などの形状操作も、見た目と当たり判定が1ミリもズレません。
- 15プリミティブ + 17演算の全てが HLSL / C# で1:1対応。

---

## サンプル (SDF Gallery)

7種類のサンプルワールドを同梱しています。**Package Manager > Samples** タブからインポートしてください。

| サンプル | 概要 | SDF式 |
|---------|------|-------|
| **Basic** | 地面 + 浮遊する球体。最もシンプルなSDFワールド。 | `min(plane, sphere)` |
| **Cosmic** | アニメーション付き太陽系 — 太陽、軌道惑星、傾斜リング、月、小惑星帯。 | `SmoothUnion(sun, planet, ring, moon, asteroids)` |
| **Fractal** | メンガーのスポンジ迷宮の内部を歩けます。ねじり変形付き。 | `Subtract(Box, Repeat(Cross))` — 1つの式で無限の複雑さ |
| **Mix** | Cosmic × Fractal 融合 — フラクタル惑星 + トーラスリング + 玉ねぎシェル。 | `SmoothUnion(Intersect(Sphere, Menger), Torus, Onion(Sphere))` |
| **DeformableWall** | 壁を触ると凹む。時間経過で回復。VRハンドインタラクション。 | `min(ground, SmoothSubtract(wall, dent_spheres...))` |
| **Mochi** | ぷにぷに餅ブロブ。掴む・合体・分裂・巨大化。SmoothUnion軟体物理。 | `SmoothUnion(ground, SmoothUnion(mochi1, mochi2, ..., k))` |
| **TerrainSculpt** | VRの手で地形を掘る・盛る。掘った穴に本当に落ちる。**SDFでしか不可能。** | `SmoothUnion(SmoothSub(plane, digs...), hills...)` |

各サンプルには以下が含まれます：
- `*_Raymarcher.shader` — SV_Depth / LOD / AO / フォグ対応レイマーチングシェーダー
- `*_Collider.cs` — UdonSharpコライダー（`#if UDONSHARP` ガード付き）
- `*.asdf.json` — Baker用の定義ファイル

### インタラクティブサンプル (VR)

**DeformableWall**、**Mochi**、**TerrainSculpt** は、VRハンドトラッキングによるリアルタイムSDF変形を実演するサンプルです。上記の静的サンプルとは異なり、毎フレーム UdonSharp から `Material.SetVectorArray` でシェーダーに動的データを送信します。

#### DeformableWall — 触って凹む壁

地面の上に立つ平面の壁。VRプレイヤーの手が壁の表面に触れると、接触点に凹みが発生し、時間の経過とともに徐々に回復します。

**仕組み:**
1. UdonSharp が `SdfBox()` による距離チェックで手の壁面への近接を検出
2. 接触時、衝撃位置とタイムスタンプを記録（循環バッファ、最大16個）
3. 毎フレーム、配列を `Material.SetVectorArray("_ImpactPoints", ...)` でシェーダーに送信
4. シェーダーが `opSmoothSubtraction(wall, sphere)` で各凹みを刻む。球の半径は指数関数的に減衰: `r = DentRadius * exp(-age * DecaySpeed)`
5. 完全に減衰した凹み（半径 < 0.005）は自動的にリサイクル

**VR操作:**
- 壁の表面に手を近づける → 凹みが出現
- 何度も叩く → 最大16個の凹みが同時に存在
- 待つ → 凹みが滑らかに元の平面に回復
- 壁に歩いて突っ込む → プレイヤーコリジョンが押し戻す

**Inspectorパラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Impact Distance | 0.08 | 凹みが発生する手と壁面の距離 |
| Impact Cooldown | 0.15秒 | 同じ手からの連続衝撃の最小間隔 |
| Dent Radius | 0.35 | 各凹みのサイズ |
| Decay Speed | 0.5 | 回復速度（大きいほど速く回復） |
| Dent Smoothness | 0.08 | SmoothSubtractionのブレンド係数 |

#### Mochi — 掴む・合体・分裂・巨大化

地面の上に置かれた柔らかい餅（もち）ブロブ。VRで掴んで引っ張って分裂させたり、押し合わせて合体させたり、どんどん大きくしたりできます。

**仕組み:**
1. 最大16個の餅球を `(位置, 半径)` のペアで管理
2. すべての餅を `opSmoothUnion` でブレンド — 近くの餅同士は自然に一体化して見える
3. 地面との接触には別の `opSmoothUnion`（低い `k` 値）を使用 — 「床にぷにっと座っている」感触
4. UdonSharp が毎フレーム `Material.SetVectorArray("_MochiData", ...)` でシェーダーに送信

**VR操作:**

| 操作 | やり方 | 結果 |
|------|--------|------|
| **掴む** | 手を餅の中に0.08秒置く | 餅が手にくっつく |
| **移動** | 掴んだまま手を動かす | 餅が手に追従する |
| **分裂** | 掴んだ位置から手を遠くに引っ張る（半径の2.5倍） | 餅が2つに分裂（体積保存: `r_new = r * cbrt(0.5)`） |
| **リリース** | さらに遠くに引っ張る（半径の4倍） | 餅が落下して地面に着地 |
| **合体** | 自由な餅同士を近づける | 1つの大きな餅に合体（`r = cbrt(r1^3 + r2^3)`） |
| **巨大化** | 合体を繰り返す | 餅がどんどん大きくなる |

**Inspectorパラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Blend K | 0.5 | 餅同士のSmoothUnion係数（大きいほど粘着） |
| Ground K | 0.15 | 地面とのSmoothUnion係数（ぷにぷに接地感） |
| Min Radius | 0.1 | 最小餅サイズ（これ以下には分裂しない） |
| Grab Threshold | 0.8 | 掴むために手が半径の何割以内に入る必要があるか |
| Grab Dwell Time | 0.08秒 | 掴み発動までの滞在時間（誤掴み防止） |
| Split Distance | 2.5 | 分裂を発動する引っ張り距離（半径の倍率） |
| Merge Threshold | 0.7 | 自動合体が発生する距離（合計半径の割合） |

#### TerrainSculpt — 掘れる・積める地形

**VRChat史上初、掘った穴に本当に落ちる体験。**

Y=0の平面地形をVRの手でリアルタイムにスカルプトできます。左手で土を盛り、右手で穴を掘る。描画もコリジョンも全く同じSDF数式で評価されるため、**掘った穴に実際に落ち、積んだ丘に実際に登れます**。

従来のVRChatでは、MeshColliderはランタイムに再計算できないため、これは原理的に不可能でした。ALICE-SDFは描画と物理の両方で同じ数式を評価するため、見た目=当たり判定が常に成立します。

**仕組み:**
1. ベース地形はY=0の地面（平面）
2. 左手が地表付近 → `opSmoothUnion(terrain, sphere)` — 手の位置に丘を追加
3. 右手が地表付近 → `opSmoothSubtraction(terrain, sphere)` — 手の位置に穴を掘削
4. 操作は循環バッファに記録（最大48個）。満杯になると最古の操作を上書き
5. UdonSharp が毎フレーム操作配列をシェーダーに送信
6. プレイヤーコリジョンも同じ数式を評価 — 穴に落ちる、丘に登れる

**VR操作:**

| 操作 | 手 | 結果 |
|------|------|------|
| **掘る** | 右手を地面に近づける | 半球状の穴が掘れる。落ちる |
| **盛る** | 左手を地面に近づける | 丘/盛り土が出現。登れる |
| **深く掘る** | 右手を穴の中に保持 | 操作ごとにさらに深く掘削 |
| **高く積む** | 左手を丘の上に保持 | さらに地形を積み上げ |

**視覚フィードバック:**
- 左手付近: 青い光（盛りモード）
- 右手付近: 赤い光（掘削モード）
- 光は手が地表面に近い時のみ表示

**地形の色分け:**
- 平面: 緑の草地
- 急斜面: 茶色の土
- 深く掘った地下: 灰色の岩石

**Inspectorパラメータ:**

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Sculpt Radius | 0.3 | スカルプトブラシのサイズ |
| Sculpt Distance | 0.15 | スカルプト発動に必要な手と地表面の距離 |
| Sculpt Cooldown | 0.12秒 | 操作間の最小間隔（バッファ溢れ防止） |
| Add Smooth | 0.25 | 丘のSmoothUnionブレンド係数（大きいほど滑らか） |
| Sub Smooth | 0.15 | 穴のSmoothSubtractionブレンド係数（大きいほど滑らかな縁） |

#### セットアップ（全インタラクティブサンプル共通）

1. シーンに **Cube** を配置（レイマーチングの描画範囲となるバウンディングボリューム）
2. Cubeを十分な大きさにスケール（例: DeformableWallなら `(12, 8, 12)`、TerrainSculptなら `(20, 10, 20)`）
3. `AliceSDF/Samples/DeformableWall`、`AliceSDF/Samples/Mochi`、または `AliceSDF/Samples/TerrainSculpt` から **マテリアル** を作成
4. CubeのMeshRendererにマテリアルを割り当て
5. 同じGameObjectに対応する `*_Collider.cs` スクリプトをアタッチ
6. VRChatで **Build & Test** — VRの手でインタラクション

**デスクトップモード:** インタラクティブ機能はVRハンドトラッキングが必要です。デスクトップモードでもSDF描画とプレイヤーコリジョンは機能しますが、スカルプト・凹み・餅操作はできません。

**マルチプレイヤー:** 全インタラクティブサンプルはローカル専用モードで動作します（各プレイヤーが独自の状態を持つ）。プレイヤー間で同期するには、データ配列に `[UdonSynced]` を付与し、状態変更時に `RequestSerialization()` を呼び出してください。

### サンプルシーンの自動生成

サンプルをインポートした後、すぐに試せるシーンを自動生成できます。

1. Unityメニュー: **ALICE-SDF > Generate Sample Scenes**
2. `Assets/AliceSDF_SampleScenes/` にシーンが生成されます
3. 任意の `SDF_*.unity` を開いて **Play** するだけ

インポート済みのサンプルを自動検出し、Camera + Light + SDFシェーダー適用済みCube + 情報UIを配置したシーンを生成します。マテリアルも `.mat` として保存されるので、インスペクタからパラメータを変更しながらリアルタイムで確認できます。

---

## クイックスタート

### ALICE-Baker (推奨)

1. **Window > ALICE-SDF Baker**
2. `.asdf.json` をドラッグ＆ドロップ（またはペースト）
3. **Bake!** → Shader + Udon + Prefab が自動生成
4. 生成されたPrefabをシーンにドラッグ＆ドロップ

### 手動セットアップ

Bakerを使わず、手書きでSDFを試したい場合の手順です。

1. **マテリアル作成**: `AliceSDF_Raymarcher.shader` を使ったマテリアルを作成。
2. **Cube配置**: シーンにCubeを置き、上記マテリアルをアタッチ（これが描画エリアになります）。
3. **コライダー設定**: 同じGameObjectに `AliceSDF_Collider.cs` をアタッチ。
4. **数式記述**:
   - シェーダー内の `map(float3 p)` 関数
   - Udonスクリプト内の `Evaluate(Vector3 p)` メソッド
   - **この2つに同じ数式を書くだけです。**

```csharp
// 例: Y=0の床と、半径1.5の球体を結合
public override float Evaluate(Vector3 p) {
    float ground = p.y;
    float sphere = (p - new Vector3(0, 1.5f, 0)).magnitude - 1.5f;
    return Mathf.Min(ground, sphere); // Union
}
```

---

## ファイル構成 (UPM)

```
com.alice.sdf/
├── package.json                     # UPMマニフェスト
├── CHANGELOG.md
├── README.md / README_JP.md
├── Runtime/
│   ├── AliceSDF.Runtime.asmdef      # Assembly Definition
│   ├── Shaders/
│   │   ├── AliceSDF_Include.cginc   # SDF関数ライブラリ (15プリミティブ + 17演算)
│   │   ├── AliceSDF_LOD.cginc       # Deep Fried 動的LOD
│   │   └── AliceSDF_Raymarcher.shader # メインレイマーチングシェーダー
│   └── Udon/
│       ├── AliceSDF_Math.cs         # ベクトル演算ヘルパー
│       ├── AliceSDF_Primitives.cs   # SDF関数 (C# — HLSL完全ミラー)
│       └── AliceSDF_Collider.cs     # プレイヤー衝突判定 + 押し戻し
├── Editor/
│   ├── AliceSDF.Editor.asmdef       # Editor Assembly Definition
│   ├── AliceSDF_Baker.cs            # Baker v0.3 (Deep Fried)
│   └── SampleSceneGenerator.cs      # メニュー: ALICE-SDF > Generate Sample Scenes
├── Samples~/                        # UPMサンプル (Package Managerからインポート)
│   └── SDF Gallery/
│       ├── SampleBasic/             # 地面 + 球体
│       ├── SampleCosmic/            # 太陽系
│       ├── SampleFractal/           # メンガーのスポンジ迷宮
│       ├── SampleMix/              # Cosmic × Fractal 融合
│       ├── SampleDeformableWall/    # インタラクティブ: 壁を触る→凹む→回復
│       ├── SampleMochi/            # インタラクティブ: 掴む・合体・分裂・巨大化
│       └── SampleTerrainSculpt/   # インタラクティブ: 掘る・積む・穴に落ちる
└── Prefabs~/                        # Unity Importから隠蔽
```

## 対応プリミティブ (53種)

| プリミティブ | HLSL | C# | 数式 |
|-------------|------|----|------|
| Sphere (球) | `sdSphere` | `Sdf.Sphere` | `length(p) - r` |
| Box (箱) | `sdBox` | `Sdf.Box` | Branchless min/max |
| Cylinder (円柱) | `sdCylinder` | `Sdf.Cylinder` | Capped vertical |
| Torus (トーラス) | `sdTorus` | `Sdf.Torus` | XZ ring |
| Plane (平面) | `sdPlane` | `Sdf.Plane` | `dot(p,n) + d` |
| Capsule (カプセル) | `sdCapsule` | `Sdf.Capsule` | Line segment + r |
| Cone (円錐) | `sdCone` | `Sdf.Cone` | Y軸キャップ付き円錐 |
| Ellipsoid (楕円体) | `sdEllipsoid` | `Sdf.Ellipsoid` | バウンド補正近似 |
| HexPrism (六角柱) | `sdHexPrism` | `Sdf.HexPrism` | Z軸六角柱 |
| Triangle (三角形) | `sdTriangle` | `Sdf.Triangle` | 3D三角形（厳密解） |
| Bezier (ベジエ曲線) | `sdBezier` | `Sdf.Bezier` | 二次ベジエ + 半径 |
| RoundedCone (丸錐) | `sdRoundedCone` | `Sdf.RoundedCone` | 滑らかキャップ付き円錐 (r1, r2) |
| Pyramid (四角錐) | `sdPyramid` | `Sdf.Pyramid` | Y軸四角錐 |
| Octahedron (八面体) | `sdOctahedron` | `Sdf.Octahedron` | 正八面体 |
| Link (鎖リンク) | `sdLink` | `Sdf.Link` | チェーンリンク (トーラス + Y伸長) |
| RoundedBox (丸角箱) | `sdRoundedBox` | `Sdf.RoundedBox` | 角丸付きボックス |
| CappedCone (切頂円錐) | `sdCappedCone` | `Sdf.CappedCone` | 円錐台 (2半径 + 高さ) |
| CappedTorus (切頂トーラス) | `sdCappedTorus` | `Sdf.CappedTorus` | トーラス弧セグメント |
| InfiniteCylinder (無限円柱) | — (inline) | `Sdf.InfiniteCylinder` | 無限円柱 (XZ平面) |
| RoundedCylinder (丸角円柱) | `sdRoundedCylinder` | `Sdf.RoundedCylinder` | 角丸付き円柱 |
| TriangularPrism (三角柱) | `sdTriangularPrism` | `Sdf.TriangularPrism` | Z軸三角柱 |
| CutSphere (切断球) | `sdCutSphere` | `Sdf.CutSphere` | 平面切断された球 |
| CutHollowSphere (中空切断球) | `sdCutHollowSphere` | `Sdf.CutHollowSphere` | 中空球の切断 |
| DeathStar (デス・スター) | `sdDeathStar` | `Sdf.DeathStar` | 球体のくり抜き |
| SolidAngle (立体角) | `sdSolidAngle` | `Sdf.SolidAngle` | 3D錐セクター |
| Rhombus (菱形) | `sdRhombus` | `Sdf.Rhombus` | 3D菱形 + 丸め |
| Horseshoe (馬蹄形) | `sdHorseshoe` | `Sdf.Horseshoe` | 馬蹄形 / アーチ |
| Vesica (ヴェシカ) | `sdVesica` | `Sdf.Vesica` | ヴェシカ・ピスキス（レンズ形） |
| InfiniteCone (無限円錐) | `sdInfiniteCone` | `Sdf.InfiniteCone` | 無限円錐 (Y軸) |
| Heart (ハート) | `sdHeart` | `Sdf.Heart` | 3Dハート (回転体) |
| Gyroid (ジャイロイド) | — (inline) | `Sdf.Gyroid` | ジャイロイド極小曲面 |
| Tube (チューブ) | `sdTube` | `Sdf.Tube` | 中空円柱 (外径, 厚さ) |
| Barrel (樽) | `sdBarrel` | `Sdf.Barrel` | 膨らみ付き樽 |
| Diamond (ダイヤモンド) | `sdDiamond` | `Sdf.Diamond` | ダイヤモンド / 双円錐 |
| ChamferedCube (面取り箱) | `sdChamferedCube` | `Sdf.ChamferedCube` | 面取り付きボックス |
| SchwarzP (シュワルツP) | — (inline) | `Sdf.SchwarzP` | シュワルツP極小曲面 |
| Superellipsoid (超楕円体) | — (inline) | `Sdf.Superellipsoid` | 球↔箱モーフ (e1, e2) |
| RoundedX (丸十字) | — (inline) | `Sdf.RoundedX` | 丸みのある十字/X形 |
| Pie (扇形) | `sdPie` | `Sdf.Pie` | セクター / 扇形 |
| Trapezoid (台形) | `sdTrapezoid` | `Sdf.Trapezoid` | 台形プリズム |
| Parallelogram (平行四辺形) | `sdParallelogram` | `Sdf.Parallelogram` | 斜め四角形プリズム |
| Tunnel (トンネル) | `sdTunnel` | `Sdf.Tunnel` | トンネル / アーチ門 |
| UnevenCapsule (非対称カプセル) | `sdUnevenCapsule` | `Sdf.UnevenCapsule` | 2半径カプセル |
| Egg (卵) | `sdEgg` | `Sdf.Egg` | 卵形 (回転体) |
| ArcShape (アーク) | `sdArcShape` | `Sdf.ArcShape` | アーチ / 橋 |
| Moon (三日月) | `sdMoon` | `Sdf.Moon` | 三日月形 |
| CrossShape (十字形) | `sdCrossShape` | `Sdf.CrossShape` | 3D十字 / プラス記号 |
| BlobbyCross (有機十字) | `sdBlobbyCross` | `Sdf.BlobbyCross` | 有機的クロス |
| ParabolaSegment (放物線) | `sdParabolaSegment` | `Sdf.ParabolaSegment` | 放物線アーチ |
| RegularPolygon (正多角形) | `sdRegularPolygon` | `Sdf.RegularPolygon` | N角形柱 |
| StarPolygon (星形) | `sdStarPolygon` | `Sdf.StarPolygon` | 星形多角形柱 |
| Stairs (階段) | `sdStairs` | `Sdf.Stairs` | 階段形状 |
| Helix (螺旋) | `sdHelix` | `Sdf.Helix` | 螺旋チューブ (バネ) |

## 対応演算 (17種)

| 演算 | HLSL | C# (Baker生成後) | 効果 |
|------|------|-------------------|------|
| 結合 | `min(a, b)` | `Mathf.Min(a, b)` | 形状を合成 |
| 交差 | `max(a, b)` | `Mathf.Max(a, b)` | 共通部分 |
| くり抜き | `max(a, -(b))` | `Mathf.Max(a, -(b))` | 形状を削る |
| 滑らか結合 | `opSmoothUnion` | インライン展開 (`inv_k` プリコンピュート) | 滑らかに合成 |
| 滑らか交差 | `opSmoothIntersection` | インライン展開 | 滑らかに交差 |
| 滑らかくり抜き | `opSmoothSubtraction` | インライン展開 | 滑らかに削る |
| 無限繰り返し | `opRepeatInfinite` | `Sdf.RepeatInfinite` | 空間タイリング |
| 有限繰り返し | `opRepeatFinite` | `Sdf.RepeatFinite` | 有限タイリング |
| 極座標繰り返し | `opPolarRepeat` | `Sdf.PolarRepeat` | Y軸円形配列 |
| ねじり | `opTwist` | `Sdf.Twist` | Y軸ねじり |
| 曲げ | `opBend` | `Sdf.Bend` | X軸曲げ |
| 角丸 | `opRound` | `Sdf.Round` | 角を丸める |
| 中空 | `opOnion` | `Sdf.Onion` | 中空シェル |
| テーパー | `opTaper` | `Sdf.Taper` | Y軸先細り |
| ディスプレイスメント | `opDisplacement` | `Sdf.Displacement` | ノイズ表面 |
| 対称 | `opSymmetry` | `Sdf.Symmetry` | 軸ミラー |
| 伸長 | `opElongate` | `Sdf.Elongate` | 軸方向伸長 |

---

## 注意事項と既知の制約

- **GPU負荷**: レイマーチングは画面を大きく覆うとGPU負荷が高まります。`AliceSDF_LOD.cginc` が自動調整しますが、巨大なオブジェクトを配置する際は注意してください。
- **鋭角な衝突**: 「Box」などの鋭い角に対して高速で衝突すると、稀にすり抜ける場合があります（Udonの更新頻度の限界）。`Push Strength` パラメータで調整可能です。
- **VRChat更新**: VRChatの仕様変更により、Udonの挙動が変わる可能性があります。

---

## ライセンス

ALICE Community License

## 作者

Moroya Sakamoto
