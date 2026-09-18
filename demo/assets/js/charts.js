/* Inline bar strips for the comparison table. Deliberately not a charting
   library: this draws one rounded rect per row and nothing else, which is
   ~30 lines against ~300KB of dependency, and it inherits the page's theme
   tokens for free because it's just CSS-styled DOM. */
(function (DEMO) {
  'use strict';
  var el = DEMO.el;

  /* value/max -> a track with a proportional fill, plus the formatted number.
     lowerIsBetter inverts the fill so the visually longer bar is always the
     better result -- otherwise a "best" MSE would draw as the shortest bar and
     read backwards at a glance. */
  function barCell(value, max, text, lowerIsBetter) {
    if (value === null || value === undefined || !isFinite(max) || max <= 0) {
      return el('span', { text: text });
    }
    var ratio = value / max;
    var shown = lowerIsBetter ? (1 - ratio) : ratio;
    shown = Math.max(0.04, Math.min(1, shown));
    return el('span.bar-cell', null, [
      el('span.mono', { text: text }),
      el('span.bar-track', null, [
        el('span.bar-fill', { style: 'width:' + (shown * 100).toFixed(1) + '%' })
      ])
    ]);
  }

  DEMO.barCell = barCell;
})(window.DEMO);
