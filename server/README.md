# ALICE-SDF REST Server

`axum` 製の薄い HTTP wrapper。ALICE-SDF の primitive 評価と operation を REST API
として公開する。`alicelaw.net/sdf-metaverse` のような cloud-served SDF UI の
backend を想定。

## ビルド & 起動

```bash
cd server
cargo run --release
# → ALICE-SDF server listening on http://0.0.0.0:8787

# カスタムポート
ALICE_SDF_SERVER_ADDR=127.0.0.1:9000 cargo run --release
```

## エンドポイント

### `GET /` / `GET /version`

```json
{ "name": "alice-sdf-server", "version": "0.1.0" }
```

### `POST /eval` — primitive 評価

```http
POST /eval
Content-Type: application/json

{
  "shape": "sphere",
  "point": [1.0, 0.0, 0.0],
  "params": { "radius": 1.0, "center": [0.0, 0.0, 0.0] }
}
```

`shape` は `"sphere" | "box" | "torus" | "cylinder"`。

レスポンス: `{ "distance": 0.0 }`

### `POST /op` — operation

```http
POST /op
Content-Type: application/json

{ "op": "smooth_union", "a": 0.5, "b": 0.6, "k": 0.1 }
```

`op` は `"union" | "intersection" | "subtraction" | "smooth_*"`。

レスポンス: `{ "distance": 0.4216 }`

## デプロイ

- **Cloud Run / Fly.io / Render**: Dockerize して push (Dockerfile は未同梱、要追加)
- **Cloudflare Workers**: wasm feature を使った別バイナリ化を検討中

## 制限事項

- 認証 / レート制限なし (本番では reverse proxy + Cloudflare で前段保護を推奨)
- メッシュ生成 / Splat / VDB エンドポイントは v0.2 で追加予定 (大きな payload のため WebSocket / streaming も検討)
- gRPC 化は将来計画 (現状 REST のみ)
