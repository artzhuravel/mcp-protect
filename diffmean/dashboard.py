#!/usr/bin/env python3
"""Simple dashboard to browse mcptox_pairs.raw.jsonl — run and open http://localhost:7331"""
import json, pathlib, http.server, urllib.parse

DATA_FILE = pathlib.Path(__file__).parent / "outputs" / "mcptox_pairs.raw.jsonl"
ROWS = [json.loads(l) for l in DATA_FILE.read_text().splitlines() if l.strip()]

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MCPTox Pairs Browser</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 20px; background: #1a1a24; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 1rem; font-weight: 600; color: #a78bfa; white-space: nowrap; }
  .filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; flex: 1; }
  select, input { background: #252535; border: 1px solid #3a3a4a; color: #e0e0e0; border-radius: 6px; padding: 4px 8px; font-size: 0.8rem; }
  input { flex: 1; min-width: 160px; }
  .count { font-size: 0.75rem; color: #666; white-space: nowrap; }
  .main { display: flex; flex: 1; overflow: hidden; }
  .list { width: 320px; min-width: 200px; border-right: 1px solid #2a2a3a; overflow-y: auto; flex-shrink: 0; }
  .item { padding: 10px 14px; border-bottom: 1px solid #1e1e2a; cursor: pointer; transition: background 0.1s; }
  .item:hover { background: #1e1e2e; }
  .item.active { background: #2a2040; border-left: 3px solid #a78bfa; }
  .item-id { font-size: 0.7rem; color: #666; font-family: monospace; }
  .item-query { font-size: 0.82rem; margin-top: 3px; color: #ccc; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .badge { display: inline-block; font-size: 0.65rem; padding: 1px 6px; border-radius: 10px; margin-top: 4px; font-weight: 500; }
  .badge-risk { background: #3d1a1a; color: #f87171; }
  .badge-server { background: #1a2d3d; color: #60a5fa; }
  .badge-paradigm { background: #1a2d1a; color: #4ade80; }
  .detail { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #444; font-size: 0.9rem; }
  .card { background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 8px; overflow: hidden; }
  .card-header { padding: 8px 14px; background: #20202e; font-size: 0.75rem; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center; }
  .card-body { padding: 12px 14px; font-size: 0.82rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word; font-family: monospace; color: #ccc; max-height: 260px; overflow-y: auto; }
  .card-body.highlight-pos { border-left: 3px solid #f87171; }
  .card-body.highlight-neg { border-left: 3px solid #4ade80; }
  .card-body.highlight-poison { border-left: 3px solid #fbbf24; }
  .meta { display: flex; gap: 8px; flex-wrap: wrap; }
  .no-results { padding: 40px; text-align: center; color: #444; }
</style>
</head>
<body>
<header>
  <h1>MCPTox Pairs</h1>
  <div class="filters">
    <input type="text" id="search" placeholder="Search query / system prompt…">
    <select id="filterServer"><option value="">All servers</option></select>
    <select id="filterRisk"><option value="">All risks</option></select>
    <select id="filterParadigm"><option value="">All paradigms</option></select>
  </div>
  <span class="count" id="count"></span>
</header>
<div class="main">
  <div class="list" id="list"></div>
  <div class="detail" id="detail"><div class="empty">Select a row to inspect</div></div>
</div>

<script>
const ROWS = __DATA__;
let filtered = ROWS;
let selected = null;

function populate(sel, key) {
  const vals = [...new Set(ROWS.map(r => r.tags[key]))].sort();
  vals.forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); });
}
populate(document.getElementById('filterServer'), 'server');
populate(document.getElementById('filterRisk'), 'security_risk');
populate(document.getElementById('filterParadigm'), 'paradigm');

function applyFilters() {
  const q = document.getElementById('search').value.toLowerCase();
  const sv = document.getElementById('filterServer').value;
  const ri = document.getElementById('filterRisk').value;
  const pa = document.getElementById('filterParadigm').value;
  filtered = ROWS.filter(r => {
    if (sv && r.tags.server !== sv) return false;
    if (ri && r.tags.security_risk !== ri) return false;
    if (pa && r.tags.paradigm !== pa) return false;
    if (q && !r.user_query.toLowerCase().includes(q) && !r.system_prompt.toLowerCase().includes(q) && !r.id.toLowerCase().includes(q)) return false;
    return true;
  });
  renderList();
}

function renderList() {
  const list = document.getElementById('list');
  document.getElementById('count').textContent = filtered.length + ' / ' + ROWS.length;
  if (!filtered.length) { list.innerHTML = '<div class="no-results">No results</div>'; return; }
  list.innerHTML = '';
  filtered.forEach((r, i) => {
    const d = document.createElement('div');
    d.className = 'item' + (r.id === selected ? ' active' : '');
    d.innerHTML = `<div class="item-id">${esc(r.id)}</div>
      <div class="item-query">${esc(r.user_query)}</div>
      <div class="meta">
        <span class="badge badge-server">${esc(r.tags.server)}</span>
        <span class="badge badge-risk">${esc(r.tags.security_risk)}</span>
        <span class="badge badge-paradigm">${esc(r.tags.paradigm)}</span>
      </div>`;
    d.onclick = () => { selected = r.id; renderList(); renderDetail(r); };
    list.appendChild(d);
  });
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function renderDetail(r) {
  const d = document.getElementById('detail');
  d.innerHTML = `
    <div class="card">
      <div class="card-header"><span>User Query</span><span class="badge badge-server">${esc(r.tags.server)}</span></div>
      <div class="card-body">${esc(r.user_query)}</div>
    </div>
    <div class="card">
      <div class="card-header"><span>Poisoned Tool</span><span class="badge badge-risk">${esc(r.tags.security_risk)}</span></div>
      <div class="card-body highlight-poison">${esc(r.extra.poisoned_tool || '—')}</div>
    </div>
    <div class="card">
      <div class="card-header"><span>y_pos — attack-compliant (bad)</span><span class="badge" style="background:#3d1a1a;color:#f87171">${esc(r.tags.y_pos_source_model || '')}</span></div>
      <div class="card-body highlight-pos">${esc(r.y_pos)}</div>
    </div>
    <div class="card">
      <div class="card-header"><span>y_neg — attack-resistant (good)</span><span class="badge" style="background:#1a2d1a;color:#4ade80">${esc(r.tags.y_neg_source_model || '')}</span></div>
      <div class="card-body highlight-neg">${esc(r.y_neg)}</div>
    </div>
    <div class="card">
      <div class="card-header">System Prompt</div>
      <div class="card-body">${esc(r.system_prompt)}</div>
    </div>`;
}

['search','filterServer','filterRisk','filterParadigm'].forEach(id => {
  document.getElementById(id).addEventListener('input', applyFilters);
});

applyFilters();
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/data.json":
            body = json.dumps(ROWS, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            page = HTML.replace("__DATA__", json.dumps(ROWS, ensure_ascii=False))
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    port = 7331
    server = http.server.HTTPServer(("localhost", port), Handler)
    print(f"Dashboard running at http://localhost:{port}  (Ctrl+C to stop)")
    server.serve_forever()
