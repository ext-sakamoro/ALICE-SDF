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

## ALICE-Baker v0.2 (Deep Fried Edition)

**「数式を書くのが難しい？」心配無用です。**

付属の **ALICE-Baker** ツールが、JSON定義ファイルから「最適化されたシェーダー」と「Udonスクリプト」を全自動生成します。

1. Unityメニューから **Window > ALICE-SDF Baker** を開く
2. `.asdf.json` をドラッグ＆ドロップ（またはJSONテキストをペースト）
3. **Bake!** ボタンをクリック

これだけで、**Prefab** が生成されます。あとはシーンに置くだけです。

### "Deep Fried" 最適化とは？

VRChatのUdon VMは、通常のC#に比べて実行速度に制約があります。ALICE-Baker v0.2は、生成されるコードに対して**狂気的な最適化（Deep Frying）**を施します。

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
- 6プリミティブ + 10演算の全てが HLSL / C# で1:1対応。

---

## サンプル (SDF Gallery)

4種類のサンプルワールドを同梱しています。**Package Manager > Samples** タブからインポートしてください。

| サンプル | 概要 | SDF式 |
|---------|------|-------|
| **Basic** | 地面 + 浮遊する球体。最もシンプルなSDFワールド。 | `min(plane, sphere)` |
| **Cosmic** | アニメーション付き太陽系 — 太陽、軌道惑星、傾斜リング、月、小惑星帯。 | `SmoothUnion(sun, planet, ring, moon, asteroids)` |
| **Fractal** | メンガーのスポンジ迷宮の内部を歩けます。ねじり変形付き。 | `Subtract(Box, Repeat(Cross))` — 1つの式で無限の複雑さ |
| **Mix** | Cosmic × Fractal 融合 — フラクタル惑星 + トーラスリング + 玉ねぎシェル。 | `SmoothUnion(Intersect(Sphere, Menger), Torus, Onion(Sphere))` |

各サンプルには以下が含まれます：
- `*_Raymarcher.shader` — SV_Depth / LOD / AO / フォグ対応レイマーチングシェーダー
- `*_Collider.cs` — UdonSharpコライダー（`#if UDONSHARP` ガード付き）
- `*.asdf.json` — Baker用の定義ファイル

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
│   │   ├── AliceSDF_Include.cginc   # SDF関数ライブラリ (6プリミティブ + 10演算)
│   │   ├── AliceSDF_LOD.cginc       # Deep Fried 動的LOD
│   │   └── AliceSDF_Raymarcher.shader # メインレイマーチングシェーダー
│   └── Udon/
│       ├── AliceSDF_Math.cs         # ベクトル演算ヘルパー
│       ├── AliceSDF_Primitives.cs   # SDF関数 (C# — HLSL完全ミラー)
│       └── AliceSDF_Collider.cs     # プレイヤー衝突判定 + 押し戻し
├── Editor/
│   ├── AliceSDF.Editor.asmdef       # Editor Assembly Definition
│   ├── AliceSDF_Baker.cs            # Baker v0.2 (Deep Fried)
│   └── SampleSceneGenerator.cs      # メニュー: ALICE-SDF > Generate Sample Scenes
├── Samples~/                        # UPMサンプル (Package Managerからインポート)
│   └── SDF Gallery/
│       ├── SampleBasic/             # 地面 + 球体
│       ├── SampleCosmic/            # 太陽系
│       ├── SampleFractal/           # メンガーのスポンジ迷宮
│       └── SampleMix/              # Cosmic × Fractal 融合
└── Prefabs~/                        # Unity Importから隠蔽
```

## 対応プリミティブ (11種)

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

## 対応演算 (16種)

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
