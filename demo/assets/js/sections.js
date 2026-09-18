/* Section renderers, keyed by manifest `type`.
 *
 * Adding a SECTION to the demo is a manifest.json edit as long as its type is
 * already here. Adding a new KIND of visual is the only case that needs code --
 * that's the intended boundary.
 *
 * Every renderer is render(bodyEl, spec, ctx) where ctx = {manifest, classes,
 * resolve(relPath), metrics}. Renderers mount into bodyEl and own their DOM.
 */
(function (DEMO) {
  'use strict';
  var el = DEMO.el, dig = DEMO.dig;

  /* ------------------------------------------------------------ formats -- */

  var FMT = {
    fixed4: function (v) { return v === null || v === undefined ? null : v.toFixed(4); },
    fixed6: function (v) { return v === null || v === undefined ? null : v.toFixed(6); },
    percent1: function (v) { return v === null || v === undefined ? null : (v * 100).toFixed(1) + '%'; },
    percent2: function (v) { return v === null || v === undefined ? null : (v * 100).toFixed(2) + '%'; },
    text: function (v) { return v === null || v === undefined ? null : String(v); }
  };

  function fmt(name, value) {
    var f = FMT[name] || FMT.text;
    return f(value);
  }

  function classById(ctx, id) {
    var found = null;
    (ctx.classes || []).forEach(function (c) { if (c.id === id) found = c; });
    return found;
  }

  /* --------------------------------------------------------------- hero -- */

  function hero(body, spec, ctx) {
    var site = ctx.manifest.site || {};
    var meta = [];
    function add(k, v) { if (v) meta.push(el('div', null, [el('span.k', { text: k }), el('span.v', { text: v })])); }
    add('Author', site.author);
    add('Degree', site.degree);
    add('University', site.university);
    add('Department', site.department);
    add('Supervision', (site.supervisors || []).join(' · '));
    add('Date', site.date);
    add('Platform', site.robot);

    var jump = (ctx.manifest.sections || [])
      .filter(function (s) { return s.nav; })
      .map(function (s) { return el('a', { href: '#' + s.id, text: s.nav }); });

    body.appendChild(el('div.hero', null, [
      el('h1', { text: site.title || 'Demo' }),
      site.subtitle ? el('p.hero-sub', { text: site.subtitle }) : null,
      meta.length ? el('div.hero-meta', null, meta) : null,
      jump.length ? el('div.hero-jump', null, jump) : null,
      site.disclaimer ? el('div.disclaimer', null, [
        el('strong', { text: 'Pre-rendered. ' }),
        document.createTextNode(site.disclaimer)
      ]) : null
    ]));
  }

  /* -------------------------------------------------------------- prose -- */

  function prose(body, spec) {
    body.appendChild(el('div.section-blurb', { html: spec.html || '' }));
  }

  /* --------------------------------------------------- comparison table -- */

  function comparison(body, spec, ctx) {
    var metrics = ctx.metrics;
    if (!metrics) { body.appendChild(el('p.note', { text: 'metrics.json not loaded.' })); return; }

    var cols = metrics.comparison.columns;
    var models = metrics.models;

    // Column maxima drive the bar widths.
    var maxima = {};
    cols.forEach(function (c) {
      var vals = models.map(function (m) { return dig(m, c.key); })
        .filter(function (v) { return typeof v === 'number'; });
      maxima[c.key] = vals.length ? Math.max.apply(null, vals) : 0;
    });

    var head = el('tr', null, [el('th', { text: 'Model' })].concat(
      cols.map(function (c) { return el('th', { text: c.label }); })
    ));

    var rows = models.map(function (m) {
      var nameCell = el('td', null, [
        el('span.model-name', { text: m.name }),
        m.winner ? el('span.badge.winner', { text: 'best' }) : null,
        m.family ? el('span.model-family', { text: m.family }) : null
      ]);
      var cells = cols.map(function (c) {
        var raw = dig(m, c.key);
        if (raw === null || raw === undefined) {
          return el('td.null', { text: c.nullText || '—' });
        }
        var text = fmt(c.format, raw);
        if (c.chart === 'bar') {
          return el('td', null, [DEMO.barCell(raw, maxima[c.key], text, !!c.lowerIsBetter)]);
        }
        return el('td.mono', { text: text });
      });
      return el('tr', null, [nameCell].concat(cells));
    });

    body.appendChild(el('div.table-wrap', null, [
      el('table.metrics', null, [
        el('thead', null, [head]),
        el('tbody', null, rows)
      ])
    ]));

    var notes = models.filter(function (m) { return m.notes; }).map(function (m) {
      return el('p.note', null, [el('strong', { text: m.name + ': ' }), document.createTextNode(m.notes)]);
    });
    notes.forEach(function (n) { body.appendChild(n); });

    body.appendChild(provenanceNote(metrics));
  }

  function provenanceNote(metrics) {
    var p = metrics.provenance || {};
    var bits = [];
    if (p.testSamples) bits.push(p.testSamples.toLocaleString() + ' held-out test samples');
    if (p.scenarios) bits.push(p.scenarios + ' scenarios');
    if (p.sharedMaxRange) bits.push('shared max_range ' + p.sharedMaxRange.toFixed(3) + ' m');
    var text = bits.join(' · ');
    var manual = p.source === 'manual';
    return el('p.note', null, [
      el('strong', { text: manual ? 'Provenance: ' : 'Measured: ' }),
      document.createTextNode(
        text + (manual
          ? ' — transcribed from a recorded run of kevin_pipeline/fair_comparison.py, not regenerated at build time.'
          : ' — regenerated at build time by ' + (p.source || 'the pipeline') + '.')
      )
    ]);
  }

  /* -------------------------------------------------------- generic table -- */

  function table(body, spec, ctx) {
    var data = spec.path ? dig(ctx.metrics, spec.path) : ctx.metrics;
    if (!data || !data.rows) { body.appendChild(el('p.note', { text: 'No table data.' })); return; }

    var head = el('tr', null, [
      el('th', { text: 'Class' }),
      el('th', { text: 'Precision' }),
      el('th', { text: 'Recall' }),
      el('th', { text: 'F1' })
    ]);

    var rows = data.rows.map(function (r) {
      var cls = classById(ctx, r.classId);
      return el('tr', null, [
        el('td', null, [
          cls ? el('span.swatch', { style: 'background:' + cls.color }) : null,
          document.createTextNode(cls ? cls.name : r.name)
        ]),
        el('td.mono', { text: fmt('percent1', r.precision) }),
        el('td.mono', { text: fmt('percent1', r.recall) }),
        el('td.mono', { text: fmt('percent1', r.f1) })
      ]);
    });

    body.appendChild(el('div.table-wrap', null, [
      el('table.metrics', null, [el('thead', null, [head]), el('tbody', null, rows)])
    ]));

    if (data.macroF1 !== undefined) {
      body.appendChild(el('p.note', null, [
        el('strong', { text: 'Macro-F1 ' + (data.macroF1 * 100).toFixed(1) + '% ' }),
        document.createTextNode(
          'across the four object classes at confidence ≥ ' + data.threshold +
          '. That threshold is a display decision, not a retrained model: it was chosen by ' +
          'sweeping 0.0–0.999 against the labelled test split, where macro-F1 peaks at ' +
          data.threshold + '. Raw argmax scores 68.1%.')
      ]));
    }
  }

  /* --------------------------------------------------------------- tabs -- */

  function tabs(body, spec, ctx) {
    var list = el('div.tablist', { role: 'tablist' });
    var panel = el('div.tabpanel');
    var mounted = {};   // index -> element, so switching back is instant
    var buttons = [];

    function select(i) {
      buttons.forEach(function (b, j) {
        b.setAttribute('aria-selected', j === i ? 'true' : 'false');
        b.tabIndex = j === i ? 0 : -1;
      });
      DEMO.clear(panel);
      if (!mounted[i]) {
        var host = el('div');
        mounted[i] = host;
        renderInto(host, spec.tabs[i], ctx);
      }
      panel.appendChild(mounted[i]);
    }

    spec.tabs.forEach(function (t, i) {
      var b = el('button.tab', {
        type: 'button', role: 'tab', 'aria-selected': 'false',
        text: t.label, onclick: function () { select(i); }
      });
      buttons.push(b);
      list.appendChild(b);
    });

    list.addEventListener('keydown', function (e) {
      var cur = buttons.findIndex(function (b) { return b.getAttribute('aria-selected') === 'true'; });
      var next = e.key === 'ArrowRight' ? cur + 1 : e.key === 'ArrowLeft' ? cur - 1 : -1;
      if (next < 0 || next >= buttons.length) return;
      e.preventDefault();
      select(next);
      buttons[next].focus();
    });

    body.appendChild(list);
    body.appendChild(panel);
    select(0);
  }

  /* ------------------------------------------------------------ gallery -- */

  function gallery(body, spec, ctx) {
    var facets = el('div.facets', { role: 'tablist' });
    var panes = el('div');
    var buttons = [];

    function show(i) {
      var item = spec.items[i];
      buttons.forEach(function (b, j) {
        b.setAttribute('aria-selected', j === i ? 'true' : 'false');
        b.tabIndex = j === i ? 0 : -1;
      });
      DEMO.clear(panes);
      var grid = el('div.panes' + (item.panes.length > 1 ? '.two' : ''));
      item.panes.forEach(function (p) {
        grid.appendChild(el('div.pane.card', null, [
          el('div.pane-label', { text: p.label }),
          el('img', { src: ctx.resolve(p.src), alt: p.label, loading: 'lazy' })
        ]));
      });
      panes.appendChild(grid);
      if (item.caption) panes.appendChild(el('p.caption', { text: item.caption }));
    }

    spec.items.forEach(function (item, i) {
      var cls = classById(ctx, item.classId);
      var b = el('button.tab', {
        type: 'button', role: 'tab', 'aria-selected': 'false',
        onclick: function () { show(i); }
      }, [
        cls ? el('span.swatch', { style: 'background:' + cls.color }) : null,
        document.createTextNode(cls ? cls.name : ('Item ' + (i + 1)))
      ]);
      buttons.push(b);
      facets.appendChild(b);
    });

    body.appendChild(facets);
    body.appendChild(panes);
    show(0);
  }

  /* -------------------------------------------------------------- video -- */

  function video(body, spec, ctx) {
    var v = el('video', {
      controls: true, preload: 'metadata', playsinline: true,
      poster: spec.poster ? ctx.resolve(spec.poster) : null
    });
    v.appendChild(el('source', { src: ctx.resolve(spec.src), type: 'video/mp4' }));

    var wrap = el('div.video-wrap.card', null, [v]);
    body.appendChild(wrap);
    if (spec.caption) body.appendChild(el('p.caption', { text: spec.caption }));

    /* The source renders were written by OpenCV as MPEG-4 Part 2 (`mp4v`),
       which no browser decodes; the build transcodes to H.264. If a
       non-transcoded file ever ships again, fail loudly and usefully rather
       than showing a black rectangle. */
    var settled = false;
    v.addEventListener('loadeddata', function () { settled = true; });
    v.addEventListener('error', showFallback);
    setTimeout(function () { if (!settled && v.readyState === 0) showFallback(); }, 6000);

    function showFallback() {
      if (wrap.querySelector('.video-fallback')) return;
      DEMO.clear(wrap);
      wrap.appendChild(el('div.video-fallback', null, [
        el('p', { text: 'This video could not be decoded by your browser.' }),
        el('p', { text: 'Use the frame-player tab for the same recording, frame by frame.' })
      ]));
    }
  }

  /* ------------------------------------------------- lazy payload-backed -- */

  function player(body, spec, ctx) {
    var host = el('div.skeleton', { text: 'Loading frames…' });
    body.appendChild(host);
    DEMO.require('player', spec.payloadId, ctx.resolve(spec.src), function (payload) {
      DEMO.clear(host);
      host.className = '';
      DEMO.createPlayer(host, payload, ctx);
    }, function () {
      host.textContent = 'Could not load this playback payload.';
    });
  }

  function pointcloud(body, spec, ctx) {
    var host = el('div.skeleton', { text: 'Loading point cloud…' });
    body.appendChild(host);
    DEMO.require('pointcloud', spec.payloadId || spec.id, ctx.resolve(spec.src), function (payload) {
      DEMO.clear(host);
      host.className = '';
      if (DEMO.createViewer) DEMO.createViewer(host, payload, ctx);
      else host.textContent = '3D viewer unavailable.';
    }, function () {
      host.textContent = 'Could not load the point-cloud payload.';
    });
  }

  function tunnel(body, spec, ctx) {
    var host = el('div.skeleton', { text: 'Loading reconstruction…' });
    body.appendChild(host);
    DEMO.require('tunnel', spec.payloadId || spec.id, ctx.resolve(spec.src), function (payload) {
      DEMO.clear(host);
      host.className = '';
      if (DEMO.createTunnel) DEMO.createTunnel(host, payload, ctx);
      else host.textContent = 'Tunnel viewer unavailable.';
    }, function () {
      host.textContent = 'Could not load the tunnel payload.';
    });
  }

  /* -------------------------------------------------------------------- */

  var RENDERERS = {
    hero: hero, prose: prose, comparison: comparison, table: table,
    tabs: tabs, gallery: gallery, video: video, player: player,
    pointcloud: pointcloud, tunnel: tunnel
  };

  function renderInto(host, spec, ctx) {
    var fn = RENDERERS[spec.type];
    if (!fn) {
      host.appendChild(el('p.note', { text: 'No renderer for section type "' + spec.type + '".' }));
      return;
    }
    fn(host, spec, ctx);
  }

  DEMO.SECTION_RENDERERS = RENDERERS;
  DEMO.renderInto = renderInto;
})(window.DEMO);
