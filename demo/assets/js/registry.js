/* Payload registry + on-demand script loader.
 *
 * Why not fetch()? This page has to work both double-clicked from a folder
 * (file://) and served over http. Under file:// the origin is opaque, so
 * fetch() and ES modules are both blocked outright -- and so is any library
 * that uses them internally (three.js's PLYLoader/PCDLoader included). Classic
 * <script> tags are not blocked, so every payload is a .js file that calls
 * DEMO.register(...) with its data. One extra line per payload, and one build
 * that works in both delivery modes with no server and no bundler.
 */
window.DEMO = (function () {
  var store = {};       // kind -> id -> payload
  var waiters = {};     // kind/id -> [callbacks]
  var requested = {};   // src -> true, so a payload is never fetched twice

  function key(kind, id) { return kind + '/' + id; }

  function register(kind, id, payload) {
    (store[kind] || (store[kind] = {}))[id] = payload;
    var k = key(kind, id);
    var queued = waiters[k];
    if (queued) {
      delete waiters[k];
      queued.forEach(function (cb) { cb(payload); });
    }
  }

  function get(kind, id) {
    return (store[kind] || {})[id];
  }

  /* Resolve `payload` for kind/id, loading `src` if it isn't registered yet.
     onError is optional; without it a failed load is reported to the console
     and the caller's callback simply never fires (the section keeps its
     skeleton, which is the honest thing to show). */
  function require(kind, id, src, cb, onError) {
    var existing = get(kind, id);
    if (existing) { cb(existing); return; }

    var k = key(kind, id);
    (waiters[k] || (waiters[k] = [])).push(cb);

    if (requested[src]) return;
    requested[src] = true;

    var s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onerror = function () {
      console.error('[demo] failed to load payload: ' + src);
      if (onError) onError(src);
    };
    document.head.appendChild(s);
  }

  return { register: register, get: get, require: require, _store: store };
})();
