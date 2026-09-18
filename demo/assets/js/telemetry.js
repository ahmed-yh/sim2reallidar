/* Telemetry formatters, keyed by the `format` string in a payload's
   telemetryFields[]. Each takes (metaEntry, fieldSpec) so a field can read
   more than one key (see "xy", which reads x and y).

   The `auto` formatter exists so a new scalar reported by a render function
   shows up in the panel with no frontend change at all -- the Python side
   emits a generic spec for keys it has no entry for. */
(function (DEMO) {
  'use strict';

  var FORMATTERS = {
    time: function (m, f) { return DEMO.formatTime(m[f.key]); },

    xy: function (m, f) {
      var keys = f.from || ['x', 'y'];
      return m[keys[0]].toFixed(1) + ', ' + m[keys[1]].toFixed(1);
    },

    meters: function (m, f) { return m[f.key].toFixed(f.digits === undefined ? 1 : f.digits) + 'm'; },

    fixed: function (m, f) { return m[f.key].toFixed(f.digits === undefined ? 4 : f.digits); },

    percent: function (m, f) {
      return (m[f.key] * 100).toFixed(f.digits === undefined ? 1 : f.digits) + '%';
    },

    text: function (m, f) { return String(m[f.key]); },

    auto: function (m, f) {
      var v = m[f.key];
      if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
      return String(v);
    }
  };

  /* A field is renderable only if every key it reads is present. The player
     filters on this before building a row, and the exporter refuses to emit a
     field the frames don't carry -- belt and braces against the failure that
     killed the real_bags/*.html pages, where the page read m.x out of a meta
     entry containing only {t, mse} and threw before playback ever started. */
  function fieldIsSatisfied(field, metaEntry) {
    var keys = field.from || [field.key];
    return keys.every(function (k) { return metaEntry[k] !== undefined; });
  }

  function formatField(field, metaEntry) {
    var fn = FORMATTERS[field.format] || FORMATTERS.auto;
    try {
      return fn(metaEntry, field);
    } catch (err) {
      return '—';
    }
  }

  DEMO.FORMATTERS = FORMATTERS;
  DEMO.fieldIsSatisfied = fieldIsSatisfied;
  DEMO.formatField = formatField;
})(window.DEMO);
