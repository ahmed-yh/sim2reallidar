/* Config-driven frame player.
 *
 * Replaces the hardcoded IIFE in viz/player_template.html. Three differences
 * that matter:
 *
 *  1. It builds its own DOM and holds element references in closures, so
 *     several players coexist on one page. The template's getElementById
 *     approach could only ever support one.
 *  2. Telemetry rows come from payload.telemetryFields, filtered against what
 *     the frames actually carry. The template hardcoded m.x / m.dist /
 *     m.class_counts, which is precisely why the real-bag pages threw on
 *     frame 0 and never started playing.
 *  3. Keyboard handling is scoped to the player, not document. With five
 *     players mounted, a document-level Space handler toggles all five.
 *
 * The real-time scheduleNext() logic is kept as-is from the template -- it was
 * correct, and it's what makes "1x" mean actual recording speed.
 */
(function (DEMO) {
  'use strict';
  var el = DEMO.el;

  var ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>';
  var ICON_PREV = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 6h2v12H7zm3 6l9 6V6z"/></svg>';
  var ICON_NEXT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6h2v12h-2zM5 18l9-6-9-6z"/></svg>';

  function createPlayer(host, payload, ctx) {
    var meta = payload.meta || [];
    var N = meta.length;
    if (!N) {
      host.appendChild(el('p.note', { text: 'This playback payload has no frames.' }));
      return null;
    }

    var frameSrc = makeFrameResolver(payload, ctx);

    /* ---- telemetry rows: only fields the data actually supports ---- */
    var fields = (payload.telemetryFields || []).filter(function (f) {
      return DEMO.fieldIsSatisfied(f, meta[0]);
    });

    var img = el('img', { alt: describeRows(payload) });
    var counter = el('span.mono', { text: '' });
    var scrub = el('input', { type: 'range', min: 0, max: N - 1, value: 0,
                              'aria-label': 'Frame' });
    var tEnd = el('span', { text: DEMO.formatTime(meta[N - 1].t) });

    var rowValues = [];
    var rowsWrap = el('div');
    fields.forEach(function (f) {
      var v = el('span.v.mono', { text: '' });
      rowValues.push({ field: f, node: v });
      rowsWrap.appendChild(el('div.player__row', null, [el('span.k', { text: f.label }), v]));
    });

    /* ---- legend, only when the payload declares one ---- */
    var legendNodes = null;
    var legendWrap = null;
    if (payload.legend && payload.legend.source) {
      legendNodes = {};
      var items = [];
      (payload.legend.classIds || []).forEach(function (cid) {
        var cls = findClass(ctx, cid);
        if (!cls) return;
        var count = el('span.player__legend-count.mono', { text: '' });
        var row = el('div.player__legend-item', null, [
          el('span.player__legend-swatch', { style: 'background:' + cls.color }),
          el('span.player__legend-name', { text: cls.name }),
          count
        ]);
        legendNodes[cid] = { row: row, count: count };
        items.push(row);
      });
      legendWrap = el('div', null,
        [el('h3', { text: payload.legend.caption || 'Classes in this frame' })].concat(items));
    }

    /* ---- transport ---- */
    var playBtn = el('button.player__play', { type: 'button', 'aria-label': 'Play or pause', html: ICON_PLAY });
    var prevBtn = el('button', { type: 'button', 'aria-label': 'Previous frame', html: ICON_PREV });
    var nextBtn = el('button', { type: 'button', 'aria-label': 'Next frame', html: ICON_NEXT });
    var speedSel = el('select', { 'aria-label': 'Playback speed' });
    [['0.5', '0.5x'], ['1', '1x (real time)'], ['2', '2x'], ['4', '4x'], ['10', '10x']]
      .forEach(function (o) {
        speedSel.appendChild(el('option', { value: o[0], text: o[1], selected: o[0] === '1' }));
      });

    var transport = el('div.player__transport', null, [
      playBtn, prevBtn, nextBtn,
      el('div.player__scrubwrap', null, [
        scrub,
        el('div.player__scrublabels', null, [el('span', { text: '0:00' }), tEnd])
      ]),
      speedSel
    ]);

    var panelChildren = [];
    if (fields.length) {
      panelChildren.push(el('div', null, [el('h3', { text: 'Telemetry' }), rowsWrap]));
    }
    if (legendWrap) panelChildren.push(legendWrap);
    var srcNote = sourceNote(payload);
    if (srcNote) panelChildren.push(srcNote);

    var root = el('div.player', { tabindex: '0', role: 'group',
                                  'aria-label': payload.title || 'Frame playback' }, [
      el('div.player__viewport', null, [
        el('div.player__labels', null, [
          el('span', { text: payload.subtitle || '' }),
          counter
        ]),
        el('div.player__imgwrap', null, [img])
      ]),
      el('div.player__panel', null, panelChildren),
      transport
    ]);
    host.appendChild(root);

    /* ---- playback engine (kept from the template) ---- */
    var current = 0, playing = false, timer = null;
    var avgDt = N > 1 ? meta[N - 1].t / (N - 1) : 1;

    function render(i) {
      current = i;
      img.src = frameSrc(i);
      counter.textContent = DEMO.pad(i + 1, 3) + ' / ' + N;
      scrub.value = i;
      var m = meta[i];

      rowValues.forEach(function (r) {
        r.node.textContent = DEMO.formatField(r.field, m);
      });

      if (legendNodes) {
        var counts = m[payload.legend.source] || {};
        Object.keys(legendNodes).forEach(function (cid) {
          var n = counts[cid] || counts[String(cid)] || 0;
          legendNodes[cid].row.classList.toggle('inactive', n === 0);
          legendNodes[cid].count.textContent = n > 0 ? n.toLocaleString() + ' pts' : '';
        });
      }
      preload(i);
    }

    var preloaded = {};
    function preload(i) {
      var ahead = (payload.frames && payload.frames.preload) || 8;
      for (var k = 1; k <= ahead; k++) {
        var j = (i + k) % N;
        if (preloaded[j]) continue;
        preloaded[j] = true;
        var im = new Image();
        im.src = frameSrc(j);
      }
    }

    function scheduleNext() {
      var next = (current + 1) % N;
      // Real elapsed time to the next frame, not a fixed step: a constant
      // interval is only "1x" by coincidence, for one particular frame count.
      var realDt = next === 0 ? avgDt : (meta[next].t - meta[current].t);
      var speed = parseFloat(speedSel.value);
      var delay = Math.max(16, (realDt / speed) * 1000);
      timer = setTimeout(function () {
        render(next);
        if (playing) scheduleNext();
      }, delay);
    }

    function play() {
      if (playing) return;
      playing = true;
      playBtn.innerHTML = ICON_PAUSE;
      scheduleNext();
    }
    function pause() {
      playing = false;
      playBtn.innerHTML = ICON_PLAY;
      clearTimeout(timer);
    }

    playBtn.addEventListener('click', function () { playing ? pause() : play(); });
    prevBtn.addEventListener('click', function () { pause(); render((current - 1 + N) % N); });
    nextBtn.addEventListener('click', function () { pause(); render((current + 1) % N); });
    scrub.addEventListener('input', function () { pause(); render(parseInt(scrub.value, 10)); });
    speedSel.addEventListener('change', function () { if (playing) { pause(); play(); } });

    root.addEventListener('keydown', function (e) {
      if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
      else if (e.code === 'ArrowLeft') { e.preventDefault(); pause(); render((current - 1 + N) % N); }
      else if (e.code === 'ArrowRight') { e.preventDefault(); pause(); render((current + 1) % N); }
    });

    /* Stop decoding frames while scrolled away or in a hidden tab. Without
       this every mounted player keeps a setTimeout chain alive for the whole
       session, which is felt on defense-room hardware. */
    var wasPlaying = false;
    DEMO.observeVisibility(root, function (visible) {
      if (visible) { if (wasPlaying) play(); }
      else { wasPlaying = playing; pause(); }
    });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) { wasPlaying = playing; pause(); }
      else if (wasPlaying) play();
    });

    render(0);
    if (!DEMO.prefersReducedMotion()) { wasPlaying = true; play(); }

    return {
      el: root, play: play, pause: pause,
      seek: function (i) { pause(); render(Math.max(0, Math.min(N - 1, i))); },
      destroy: function () { pause(); if (root.parentNode) root.parentNode.removeChild(root); }
    };
  }

  /* files mode -> payloads/players/frames/<id>/0000.png
     inline mode -> data: URI from the payload itself */
  function makeFrameResolver(payload, ctx) {
    var f = payload.frames || {};
    if (f.mode === 'inline') {
      var data = f.data || [];
      var mime = 'image/' + (f.format || 'png');
      return function (i) { return 'data:' + mime + ';base64,' + data[i]; };
    }
    var base = ctx.resolve(f.base || '');
    var ext = '.' + (f.format || 'png');
    var pad = f.pad || 4;
    return function (i) { return base + '/' + DEMO.pad(i, pad) + ext; };
  }

  function describeRows(payload) {
    var rows = (payload.rows || []).map(function (r) { return r.label; });
    return rows.length ? ('LiDAR range image: ' + rows.join(', ')) : 'LiDAR range image';
  }

  function findClass(ctx, id) {
    var found = null;
    (ctx.classes || []).forEach(function (c) { if (c.id === id) found = c; });
    return found;
  }

  /* Provenance line: which checkpoint produced these frames, and -- for real
     data -- the confidence threshold, which is a display decision and has to
     be stated or it becomes the question at the defense. */
  function sourceNote(payload) {
    var s = payload.source || {};
    var bits = [];
    if (s.checkpoint) bits.push('Checkpoint <code>' + s.checkpoint + '</code>');
    if (s.bag) bits.push('Bag <code>' + s.bag + '</code>');
    if (s.classConfThreshold) {
      bits.push('Predictions shown at confidence ≥ ' + s.classConfThreshold +
                ' (display threshold, not a retrained model)');
    }
    if (s.nFrames && s.nRawAvailable && s.nFrames < s.nRawAvailable) {
      bits.push(s.nFrames + ' of ' + s.nRawAvailable + ' scans, evenly sampled');
    }
    if (!bits.length) return null;
    return DEMO.el('div.player__source', { html: bits.join('<br>') });
  }

  DEMO.createPlayer = createPlayer;
})(window.DEMO);
