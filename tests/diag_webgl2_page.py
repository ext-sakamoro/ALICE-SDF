"""Measurement only (diagnostic branch): writes a page that compiles one WebGL2 fragment shader
and draws one frame, with the times in the page text (read back with --dump-dom)

usage: diag_webgl2_page.py <shader.frag> <out.html>
"""

import json
import sys

VERT = "#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n"

PAGE = """<!doctype html><html><head><meta charset="utf-8"></head><body><canvas id="c" width="64" height="64"></canvas><pre id="r">not run</pre>
<script>
const out = (s) => { document.getElementById('r').textContent = s; };
try {
  const gl = document.getElementById('c').getContext('webgl2');
  if (!gl) throw new Error('no webgl2 context');
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const t0 = performance.now();
  const vs = sh(gl.VERTEX_SHADER, VERT);
  const fs = sh(gl.FRAGMENT_SHADER, FRAG);
  const p = gl.createProgram();
  gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
  const ok = gl.getProgramParameter(p, gl.LINK_STATUS);
  const t1 = performance.now();
  if (!ok) throw new Error('link failed: ' + gl.getShaderInfoLog(fs) + ' ' + gl.getProgramInfoLog(p));
  gl.useProgram(p);
  const u = (n) => gl.getUniformLocation(p, n);
  gl.uniform2f(u('uRes'), 64, 64);
  gl.uniform3f(u('uCamPos'), 0.0, 0.1, 1.0);
  gl.uniform3f(u('uCamFwd'), 0.0, 0.0, -1.0);
  gl.uniform3f(u('uCamRight'), 1.0, 0.0, 0.0);
  gl.uniform3f(u('uCamUp'), 0.0, 1.0, 0.0);
  gl.uniform1f(u('uMaxDist'), 10.0);
  gl.uniform1f(u('uDayPhase'), 0.5);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  const px = new Uint8Array(4);
  gl.readPixels(32, 32, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const t2 = performance.now();
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.readPixels(32, 32, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const t3 = performance.now();
  out('WEBGL2 renderer=' + renderer + ' link_s=' + ((t1 - t0) / 1000).toFixed(2)
    + ' first_draw_s=' + ((t2 - t1) / 1000).toFixed(2) + ' second_draw_s=' + ((t3 - t2) / 1000).toFixed(2)
    + ' pixel=' + Array.from(px).join(','));
} catch (e) {
  out('WEBGL2 error ' + e.message);
}
</script></body></html>
"""


def main() -> None:
    frag_path, out_path = sys.argv[1], sys.argv[2]
    with open(frag_path, encoding="utf-8") as f:
        frag = f.read()
    script = "const VERT = " + json.dumps(VERT) + ";\nconst FRAG = " + json.dumps(frag) + ";\n"
    page = PAGE.replace("<script>\n", "<script>\n" + script, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
