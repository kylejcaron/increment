/* Native HTML adapter for the dashboard shell.
 *
 * Python-rendered sections (CoefTable results, health, provenance, metric detail, explore plots)
 * arrive as HTML strings. The adapter mounts them into the shell without leaking into it:
 *   - every id is made unique per mount, and every reference to it (HTML attributes, ARIA lists,
 *     SVG href / url(#...) paint and clip references, inline styles, CSS selectors) is remapped;
 *   - the embedded <style> rules are parsed by the browser's CSS engine and re-emitted scoped to
 *     the mount host, never by text splitting;
 *   - display aliases for native column and disclosure titles are applied;
 *   - metric-label cells that carry data-inc-metric="<metric key>" become labelled controls.
 *     The key is always the renderer's explicit attribute, never recovered from rendered text.
 *
 * createNativeAdapter({ metricByKey, el, str, own }) -> { mountNative, planBands, keepColumns }
 *   mountNative(host, html, { metrics: false | 'label' | 'select' })   host.id is the id prefix.
 *   planBands(host, available) / keepColumns(host, columns) split a table that is too wide for a
 *   page into column bands (identity columns repeated); see their comments.
 *
 * This file is embedded verbatim in a classic <script> before the shell code, so it declares one
 * global function and must never contain a closing script tag or an HTML comment opener.
 */
function createNativeAdapter(dependencies) {
  'use strict';
  const { metricByKey, el, str, own } = dependencies;

  const REF_LISTS = ['headers', 'aria-labelledby', 'aria-describedby', 'aria-controls', 'aria-owns', 'aria-flowto', 'aria-details'];
  const REF_SINGLE = ['for', 'list', 'form', 'aria-activedescendant', 'aria-errormessage'];
  const HREF_ATTRS = ['href', 'xlink:href', 'usemap'];
  const STRING = '"(?:\\\\.|[^"\\\\])*"|\'(?:\\\\.|[^\'\\\\])*\'';
  const URL_REF = new RegExp('(' + STRING + ')|url\\(\\s*(["\']?)#([^"\')\\s]+)\\2\\s*\\)', 'g');
  const ATTR_SELECTOR = new RegExp(
    '^\\s*((?:\\\\.|[^\\s~|^$*=\\]\\\\])+)\\s*(?:([~|^$*]?=)\\s*(' + STRING + '|[^\\s"\']+)\\s*([iIsS])?)?\\s*$');

  /* ---------- Identifiers ---------------------------------------------------- */

  function outerSvg(node) {
    let found = null;
    for (let n = node; n; n = n.parentElement) if (n.localName === 'svg') found = n;
    return found;
  }

  /**
   * Make every id under `root` unique to `prefix` and remap every reference to it.
   * SVG ids are scoped to their outermost <svg> (nested SVG shares its outer scope), so identical
   * chart fragments that reuse ids such as `clip` keep pointing at their own definitions.
   * Returns `resolver(node)` -> (oldId) => newId | undefined, for references made from `node`.
   */
  function remapIds(root, prefix) {
    const elements = [...root.querySelectorAll('*')];
    const svgNumbers = new Map();
    const svgScopes = new Map();
    const htmlIds = new Map();
    const svgIds = new Map();
    const assigned = new Map();
    const used = new Set();
    for (const node of elements) {
      const old = node.getAttribute('id');
      if (!old) continue;
      const svg = outerSvg(node);
      let scope = '';
      if (svg) {
        if (!svgNumbers.has(svg)) svgNumbers.set(svg, svgNumbers.size);
        scope = 's' + svgNumbers.get(svg) + '_';
      }
      const base = prefix + '__' + scope + old;
      let candidate = base;
      for (let n = 1; used.has(candidate); ) candidate = base + '_' + ++n;
      used.add(candidate);
      assigned.set(node, candidate);
      if (svg) {
        if (!svgScopes.has(svg)) svgScopes.set(svg, new Map());
        if (!svgScopes.get(svg).has(old)) svgScopes.get(svg).set(old, candidate);
        if (!svgIds.has(old)) svgIds.set(old, candidate);
      } else if (!htmlIds.has(old)) {
        htmlIds.set(old, candidate);
      }
    }
    // A reference prefers its own SVG's definition, then the document's, then any other SVG's.
    const resolver = (node) => {
      const svg = node ? outerSvg(node) : null;
      const local = svg ? svgScopes.get(svg) : null;
      return (old) => (local && local.get(old)) || htmlIds.get(old) || svgIds.get(old);
    };
    for (const node of elements) {
      const resolve = resolver(node);
      if (assigned.has(node)) node.setAttribute('id', assigned.get(node));
      for (const attribute of [...node.attributes]) {
        const name = attribute.name;
        const value = attribute.value;
        if (name === 'id') continue;
        let next = value;
        if (REF_LISTS.includes(name)) {
          next = value.split(/\s+/).filter(Boolean).map((token) => resolve(token) || token).join(' ');
        } else if (REF_SINGLE.includes(name)) {
          next = resolve(value.trim()) || value;
        } else if (HREF_ATTRS.includes(name) && value.charAt(0) === '#') {
          const target = resolve(value.slice(1));
          if (target) next = '#' + target;
        } else if (value.indexOf('url(') !== -1) {
          next = rewriteUrlRefs(value, resolve);
        }
        if (next !== value) node.setAttribute(name, next);
      }
    }
    return resolver;
  }

  /** Remap url(#id) references; quoted strings (for example generated content) are left alone. */
  function rewriteUrlRefs(text, resolve) {
    return text.replace(URL_REF, (match, quoted, quote, id) => {
      if (quoted) return match;
      const target = resolve(id);
      return target ? 'url(#' + target + ')' : match;
    });
  }

  /* ---------- CSS scoping ---------------------------------------------------- */

  const cssIdent = (value) =>
    typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(value) : value.replace(/[^\w-]/g, (c) => '\\' + c);

  /** Index just past the string that opens at `start`. */
  function skipString(text, start) {
    const quote = text[start];
    let i = start + 1;
    while (i < text.length && text[i] !== quote) i += text[i] === '\\' ? 2 : 1;
    return Math.min(i + 1, text.length);
  }

  /** Index just past the attribute selector that opens at `start`, honouring strings. */
  function skipBrackets(text, start) {
    let i = start + 1;
    while (i < text.length && text[i] !== ']') {
      if (text[i] === '"' || text[i] === "'") i = skipString(text, i);
      else i += text[i] === '\\' ? 2 : 1;
    }
    return Math.min(i + 1, text.length);
  }

  /** The identifier starting at `start`: its raw end index and its unescaped value. */
  function readIdent(text, start) {
    let i = start;
    let value = '';
    while (i < text.length) {
      const ch = text[i];
      if (ch === '\\') {
        const hex = /^[0-9a-fA-F]{1,6}\s?/.exec(text.slice(i + 1));
        if (hex) {
          value += String.fromCodePoint(Math.min(parseInt(hex[0], 16), 0x10ffff) || 0xfffd);
          i += 1 + hex[0].length;
        } else if (i + 1 < text.length) {
          value += text[i + 1];
          i += 2;
        } else {
          i += 1;
        }
      } else if (/[\w-]/.test(ch) || ch.charCodeAt(0) >= 128) {
        value += ch;
        i += 1;
      } else {
        break;
      }
    }
    return { end: i, value };
  }

  /** Map the ids a reference-bearing attribute value points at. */
  function mapAttributeValue(name, value, mapId) {
    if (name === 'id' || REF_SINGLE.includes(name)) return mapId(value.trim()) || value;
    if (REF_LISTS.includes(name)) return value.split(/\s+/).filter(Boolean).map((t) => mapId(t) || t).join(' ');
    if (/^(style|clip-path|mask|filter|fill|stroke|marker-start|marker-mid|marker-end)$/.test(name) && value.indexOf('url(') !== -1) return rewriteUrlRefs(value, mapId);
    if (HREF_ATTRS.includes(name) && value.charAt(0) === '#') {
      const target = mapId(value.slice(1));
      return target ? '#' + target : value;
    }
    return value;
  }

  /** Remap the target of one attribute selector such as [for="id"] or [aria-labelledby~=id]. */
  function mapAttributeSelector(text, mapId) {
    const parts = ATTR_SELECTOR.exec(text.slice(1, -1));
    if (!parts || !parts[2] || (parts[2] !== '=' && parts[2] !== '~=')) return text;
    const name = parts[1].replace(/^.*\|/, '').toLowerCase();
    const quoted = /^["']/.test(parts[3]);
    const raw = quoted ? parts[3].slice(1, -1) : parts[3];
    const value = raw.replace(/\\([0-9a-fA-F]{1,6}\s?|.)/g, (m, c) => (/^[0-9a-fA-F]/.test(c) ? String.fromCodePoint(parseInt(c, 16) || 0xfffd) : c));
    const next = mapAttributeValue(name, value, mapId);
    if (next === value) return text;
    return '[' + parts[1] + parts[2] + '"' + next.replace(/["\\]/g, '\\$&') + '"' + (parts[4] ? ' ' + parts[4] : '') + ']';
  }

  /** Remap #id tokens and id-bearing attribute selectors; strings and other attributes are inert. */
  function mapSelectorIds(selector, mapId) {
    let out = '';
    let i = 0;
    while (i < selector.length) {
      const ch = selector[i];
      if (ch === '\\') {
        out += selector.slice(i, i + 2);
        i += 2;
      } else if (ch === '"' || ch === "'") {
        const end = skipString(selector, i);
        out += selector.slice(i, end);
        i = end;
      } else if (ch === '[') {
        const end = skipBrackets(selector, i);
        out += mapAttributeSelector(selector.slice(i, end), mapId);
        i = end;
      } else if (ch === '#') {
        const ident = readIdent(selector, i + 1);
        const target = ident.end > i + 1 ? mapId(ident.value) : undefined;
        out += target ? '#' + cssIdent(target) : selector.slice(i, ident.end);
        i = ident.end;
      } else {
        out += ch;
        i += 1;
      }
    }
    return out;
  }

  /** Split a selector list at its top-level commas; :is(a, b) and [x="a,b"] stay whole. */
  function splitSelectorList(selector) {
    const parts = [];
    let depth = 0;
    let start = 0;
    let i = 0;
    while (i < selector.length) {
      const ch = selector[i];
      if (ch === '\\') i += 2;
      else if (ch === '"' || ch === "'") i = skipString(selector, i);
      else if (ch === '[') i = skipBrackets(selector, i);
      else {
        if (ch === '(') depth += 1;
        else if (ch === ')') depth = Math.max(0, depth - 1);
        else if (ch === ',' && depth === 0) {
          parts.push(selector.slice(start, i));
          start = i + 1;
        }
        i += 1;
      }
    }
    parts.push(selector.slice(start));
    return parts.map((part) => part.trim()).filter(Boolean);
  }

  /** Prefix every selector of a top-level rule with the mount host. */
  function scopeSelectorList(selector, context) {
    const host = '#' + cssIdent(context.hostId);
    return splitSelectorList(mapSelectorIds(selector, context.mapId)).map((next) => {
      if (next.startsWith(host + '__')) return next;
      const root = /^(:root|html|body)(?![\w-])/.exec(next);
      return root ? host + next.slice(root[0].length) : host + ' ' + next;
    }).join(', ');
  }

  /** Re-emit one parsed rule scoped to the host. Unscopable rule kinds are dropped, never leaked. */
  function scopeRule(rule, context, nested) {
    if (typeof CSSImportRule !== 'undefined' && rule instanceof CSSImportRule) return '';
    if (typeof CSSFontFaceRule !== 'undefined' && rule instanceof CSSFontFaceRule) return '';
    if (rule instanceof CSSStyleRule) {
      // Nested rules are relative to their already-scoped parent: only their ids are remapped.
      const selector = nested ? mapSelectorIds(rule.selectorText, context.mapId) : scopeSelectorList(rule.selectorText, context);
      if (!selector) return '';
      const declarations = rewriteUrlRefs(rule.style.cssText, context.mapId);
      const inner = rule.cssRules ? scopeRules(rule.cssRules, context, true) : '';
      return selector + ' { ' + declarations + ' ' + inner + ' }';
    }
    if (typeof CSSGroupingRule !== 'undefined' && rule instanceof CSSGroupingRule) {
      const text = rule.cssText;
      const header = text.slice(0, text.indexOf('{')).trim();
      return header + ' { ' + scopeRules(rule.cssRules, context, nested) + ' }';
    }
    return rewriteUrlRefs(rule.cssText, context.mapId);
  }

  function scopeRules(rules, context, nested) {
    return [...rules].map((rule) => scopeRule(rule, context, nested)).join('\n');
  }

  /** Parse with the browser's CSS engine; nothing is fetched (@import is ignored or dropped). */
  function parseSheet(css) {
    try {
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(css);
      return sheet;
    } catch (error) { /* no constructable stylesheets: parse in an inert document instead */ }
    try {
      const doc = document.implementation.createHTMLDocument('');
      const style = doc.createElement('style');
      style.textContent = css;
      doc.head.append(style);
      return style.sheet;
    } catch (error) {
      return null;
    }
  }

  function rewriteCss(css, context) {
    const sheet = parseSheet(css);
    return sheet ? scopeRules(sheet.cssRules, context, false) : '';
  }

  /* ---------- Titles and tables ---------------------------------------------- */

  // Idempotent display aliases for native column and disclosure titles.
  const HEADER_ALIASES = { metric: 'Metric', group_id: 'Arm', segment: 'Segment', 'Lift %': 'Relative lift', 'Lift Plot': 'Effect interval' };
  const SUMMARY_ALIASES = { 'Evidence geometry': 'How to read these intervals', 'Statistical interpretation': 'Methods & assumptions' };

  /** Replace the title text, keeping any structure the title sits in. */
  function retitle(node, original, alias) {
    if (!node.children.length) { node.textContent = alias; return; }
    for (const leaf of node.querySelectorAll('*')) {
      if (!leaf.children.length && leaf.textContent.trim() === original) { leaf.textContent = alias; return; }
    }
  }

  /** The table's cells with their grid position, honouring colspan and rowspan. */
  function gridOf(table) {
    const occupied = [];
    const cells = [];
    let columns = 0;
    [...table.rows].forEach((tr, r) => {
      occupied[r] = occupied[r] || new Set();
      let c = 0;
      for (const cell of tr.cells) {
        while (occupied[r].has(c)) c += 1;
        const colspan = Math.max(1, cell.colSpan | 0);
        const rowspan = Math.max(1, cell.rowSpan | 0);
        cells.push({ cell, row: r, col: c, colspan, inHead: tr.parentElement === table.tHead });
        for (let dr = 0; dr < rowspan; dr += 1) {
          occupied[r + dr] = occupied[r + dr] || new Set();
          for (let dc = 0; dc < colspan; dc += 1) occupied[r + dr].add(c + dc);
        }
        c += colspan;
        columns = Math.max(columns, c);
      }
    });
    return { cells, columns };
  }

  /** Stamp the grid width the print layout keys on; spanned headers mark mixed-method tables. */
  function describeTable(table) {
    const grid = gridOf(table);
    if (grid.columns) table.dataset.cols = String(grid.columns);
    if (grid.cells.some((entry) => entry.inHead && entry.colspan > 1)) table.dataset.spanned = '';
    else delete table.dataset.spanned;
  }

  /* ---------- Metric controls ------------------------------------------------ */

  /** Display labels can include renderer qualifiers; identity always comes from the explicit mark. */
  function metricControl(metric, interactive, label) {
    const control = el(interactive ? 'button' : 'span', { className: 'metric-select' },
      el('strong', { text: label }),
      label !== metric.key ? el('small', { text: metric.key }) : null);
    if (interactive) {
      control.type = 'button';
      control.dataset.metric = metric.key;
      control.setAttribute('aria-pressed', 'false');
      control.setAttribute('aria-controls', 'inspector');
    }
    return control;
  }

  /**
   * Turn every renderer-marked metric label into a control. The renderer states the key with
   * data-inc-metric on the label cell or on an element inside it. A mark on the cell replaces the
   * cell content; a mark on an inner element replaces only that element, keeping siblings
   * (method badges, notes). Unknown keys and unmarked cells are left exactly as rendered.
   */
  function labelMetricCells(host, interactive) {
    for (const mark of [...host.querySelectorAll('tbody [data-inc-metric]')]) {
      if (!mark.isConnected) continue;
      const outer = mark.parentElement && mark.parentElement.closest('[data-inc-metric]');
      if (outer && host.contains(outer)) continue;
      const cell = mark.closest('td, th');
      if (!cell || !host.contains(cell) || cell.classList.contains('gt_group_heading')) continue;
      const metric = metricByKey.get(mark.getAttribute('data-inc-metric'));
      if (!metric) continue;
      const rendered = mark.textContent.trim();
      const label = rendered === metric.key ? str(metric.label) || metric.key : rendered || str(metric.label) || metric.key;
      const control = metricControl(metric, interactive, label);
      if (mark === cell) cell.replaceChildren(control);
      else mark.replaceWith(control);
      if (interactive) {
        const row = cell.closest('tr');
        row.dataset.metric = metric.key;
        row.classList.add('metric-row');
      }
    }
  }

  /* ---------- Mounting ------------------------------------------------------- */

  /**
   * Mount Python-rendered HTML into `host` (its id is the id prefix, so it must be unique).
   * options.metrics: false | 'label' | 'select' controls how marked metric cells render.
   */
  function mountNative(host, html, options) {
    const settings = options || {};
    const template = document.createElement('template');
    template.innerHTML = str(html);
    const fragment = template.content;
    for (const script of fragment.querySelectorAll('script')) script.remove();
    const resolver = remapIds(fragment, host.id);
    for (const style of fragment.querySelectorAll('style')) {
      const resolve = resolver(style);
      style.textContent = rewriteCss(style.textContent, { hostId: host.id, mapId: resolve });
    }
    host.replaceChildren(fragment);
    for (const th of host.querySelectorAll('thead th:not([data-inc-literal])')) {
      const original = th.textContent.trim();
      const alias = own(HEADER_ALIASES, original);
      if (alias) retitle(th, original, alias);
    }
    for (const summary of host.querySelectorAll('summary')) {
      const original = summary.textContent.trim();
      const alias = own(SUMMARY_ALIASES, original);
      if (alias) retitle(summary, original, alias);
    }
    for (const table of host.querySelectorAll('table')) describeTable(table);
    if (settings.metrics) labelMetricCells(host, settings.metrics === 'select');
    return host;
  }

  /* ---------- Column bands (tables wider than a page) ------------------------ */

  const IDENTITY_HEADER = /^(metric|arm|segment)s?$/i;

  /**
   * Split the first table of `host` into column bands that each fit `available` pixels. The
   * leading identity columns (Metric, Arm, Segment; at least the first column) repeat in every
   * band; the remaining columns move as whole units when their method spanner fits, or separately
   * when it does not. Call while the table is laid out at its natural width so column widths are
   * the real ones. Returns null when the table already fits `available` (chart columns count at
   * 70% of their natural width, since they scale) or has nothing to split, else an array of
   * column index lists (one per band) for keepColumns.
   */
  function planBands(host, available) {
    const table = host.querySelector('table');
    if (!table) return null;
    const { cells, columns } = gridOf(table);
    if (columns < 3) return null;
    const widths = new Array(columns).fill(0);
    const charts = new Set();
    for (const entry of cells) {
      if (entry.colspan !== 1) continue;
      widths[entry.col] = Math.max(widths[entry.col], entry.cell.getBoundingClientRect().width);
      if (entry.cell.querySelector('svg')) charts.add(entry.col);
    }
    for (const col of charts) widths[col] *= 0.7;
    const headTop = (col) => cells.find((entry) => entry.inHead && entry.col <= col && col < entry.col + entry.colspan);
    let fixed = 0;
    while (fixed < columns - 1) {
      const head = headTop(fixed);
      if (head && head.colspan === 1 && !head.cell.hasAttribute('data-inc-literal') && IDENTITY_HEADER.test(head.cell.textContent.trim())) fixed += 1;
      else break;
    }
    fixed = Math.max(1, fixed);
    const units = [];
    for (let col = fixed; col < columns; ) {
      const head = headTop(col);
      const span = head && head.col === col ? head.colspan : 1;
      units.push({ start: col, end: Math.min(columns, col + span) });
      col += span;
    }
    const sum = (start, end) => widths.slice(start, end).reduce((a, b) => a + b, 0);
    // The identity columns wrap long labels, so they never claim more than a third of the page.
    const fixedWidth = Math.min(sum(0, fixed), available * 0.34);
    if (fixedWidth + sum(fixed, columns) <= available) return null;
    // Preserve a method spanner when it fits; repeat it across smaller bands when it cannot.
    const printable = units.flatMap((unit) => sum(unit.start, unit.end) <= available - fixedWidth
      ? [unit]
      : Array.from({ length: unit.end - unit.start }, (_, offset) => ({
        start: unit.start + offset, end: unit.start + offset + 1
      })));
    if (printable.length < 2) return null;
    const identity = Array.from({ length: fixed }, (_, index) => index);
    const bands = [];
    let current = [];
    let used = fixedWidth;
    for (const unit of printable) {
      const width = sum(unit.start, unit.end);
      if (current.length && used + width > available) {
        bands.push(current);
        current = [];
        used = fixedWidth;
      }
      for (let col = unit.start; col < unit.end; col += 1) current.push(col);
      used += width;
    }
    bands.push(current);
    return bands.length < 2 ? null : bands.map((band) => identity.concat(band));
  }

  /** Keep only `columns` in the first table of `host`; spans shrink, emptied cells disappear. */
  function keepColumns(host, columns) {
    const table = host.querySelector('table');
    if (!table) return;
    const keep = new Set(columns);
    const { cells } = gridOf(table);
    for (const entry of cells) {
      let remaining = 0;
      for (let col = entry.col; col < entry.col + entry.colspan; col += 1) if (keep.has(col)) remaining += 1;
      if (remaining === 0) entry.cell.remove();
      else if (remaining !== entry.colspan) entry.cell.colSpan = remaining;
    }
    for (const row of [...table.rows]) if (!row.cells.length) row.remove();
    describeTable(table);
  }

  return { mountNative, planBands, keepColumns };
}
