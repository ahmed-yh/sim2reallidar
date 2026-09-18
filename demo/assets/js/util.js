/* Small DOM + formatting helpers shared by every renderer. No dependencies. */
(function (DEMO) {
  'use strict';

  /* el('div.card', {id: 'x'}, [child, 'text']) */
  function el(spec, attrs, children) {
    var parts = String(spec).split('.');
    var tagAndId = parts.shift();
    var hashIdx = tagAndId.indexOf('#');
    var tag = tagAndId, id = null;
    if (hashIdx >= 0) { tag = tagAndId.slice(0, hashIdx); id = tagAndId.slice(hashIdx + 1); }
    var node = document.createElement(tag || 'div');
    if (id) node.id = id;
    if (parts.length) node.className = parts.join(' ');

    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k === 'text') { node.textContent = v; }
        else if (k === 'html') { node.innerHTML = v; }
        else if (k === 'class') { node.className = node.className ? node.className + ' ' + v : v; }
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') {
          node.addEventListener(k.slice(2).toLowerCase(), v);
        } else { node.setAttribute(k, v === true ? '' : v); }
      });
    }

    (children || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function pad(n, width) {
    var s = String(n);
    while (s.length < width) s = '0' + s;
    return s;
  }

  function formatTime(seconds) {
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ':' + pad(s, 2);
  }

  /* Run cb once, shortly before the element scrolls into view. Used to mount
     sections lazily so the page never holds every payload at once. Degrades to
     mounting immediately where IntersectionObserver is unavailable. */
  function onVisible(node, options, cb) {
    if (typeof IntersectionObserver !== 'function') { cb(); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { io.disconnect(); cb(); }
      });
    }, { rootMargin: (options && options.rootMargin) || '300px' });
    io.observe(node);
  }

  /* Fires cb(true/false) as the element enters/leaves view, and keeps firing.
     Players and the 3D viewers use this to stop work while off-screen. */
  function observeVisibility(node, cb) {
    if (typeof IntersectionObserver !== 'function') { cb(true); return function () {}; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { cb(e.isIntersecting); });
    }, { rootMargin: '0px' });
    io.observe(node);
    return function () { io.disconnect(); };
  }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /* Read "metrics.mse" out of {metrics: {mse: 1}} */
  function dig(obj, path) {
    return String(path).split('.').reduce(function (o, k) {
      return (o === null || o === undefined) ? undefined : o[k];
    }, obj);
  }

  /* base64 -> Uint8Array, for the geometry payloads. atob is available in
     every browser this targets and needs no polyfill. */
  function b64ToBytes(b64) {
    var bin = atob(b64);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  DEMO.el = el;
  DEMO.clear = clear;
  DEMO.pad = pad;
  DEMO.formatTime = formatTime;
  DEMO.onVisible = onVisible;
  DEMO.observeVisibility = observeVisibility;
  DEMO.prefersReducedMotion = prefersReducedMotion;
  DEMO.dig = dig;
  DEMO.b64ToBytes = b64ToBytes;
})(window.DEMO);
