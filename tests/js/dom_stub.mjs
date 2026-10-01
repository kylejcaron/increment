// Minimal, hand-rolled DOM stub covering exactly the browser API surface
// docs/javascripts/compatibility.js uses: id/class/attribute-presence
// selectors, `.dataset`, `.value`, `.className` (get/set),
// `.classList.add`/`.classList.contains`, `addEventListener`,
// `setAttribute` / `getAttribute` / `removeAttribute`, `textContent`,
// `append`, `replaceChildren`, and `document.createElement`. No jsdom or
// other npm dependency -- this project has none, and the real script's
// DOM usage is small and closed enough to stub directly with Node's
// built-ins.

class FakeElement {
  constructor(tagName, { id, className } = {}) {
    this.tagName = tagName;
    this.attributes = new Map();
    this.children = [];
    this.listeners = {};
    this._text = "";
    this._value = "";
    if (id !== undefined) this.setAttribute("id", id);
    if (className !== undefined) this.setAttribute("class", className);
    this.dataset = new Proxy(
      {},
      {
        get: (_target, key) => this.attributes.get(`data-${String(key)}`),
      }
    );
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  addEventListener(type, handler) {
    (this.listeners[type] ??= []).push(handler);
  }

  dispatch(type) {
    for (const handler of this.listeners[type] ?? []) handler();
  }

  append(...nodes) {
    this.children.push(...nodes);
  }

  replaceChildren() {
    this.children = [];
  }

  get id() {
    return this.getAttribute("id") ?? "";
  }

  get className() {
    return this.getAttribute("class") ?? "";
  }

  set className(value) {
    this.setAttribute("class", value);
  }

  get classList() {
    const self = this;
    const names = () => self.className.split(/\s+/).filter(Boolean);
    return {
      add: (...toAdd) => {
        const current = new Set(names());
        for (const name of toAdd) current.add(name);
        self.setAttribute("class", [...current].join(" "));
      },
      remove: (...toRemove) => {
        const current = new Set(names());
        for (const name of toRemove) current.delete(name);
        self.setAttribute("class", [...current].join(" "));
      },
      contains: (name) => names().includes(name),
    };
  }

  get value() {
    return this._value;
  }

  set value(v) {
    this._value = v;
  }

  get textContent() {
    if (this.children.length) return this.children.map((child) => child.textContent).join("");
    return this._text;
  }

  set textContent(value) {
    this.children = [];
    this._text = value;
  }

  querySelectorAll(selector) {
    const out = [];
    collect(this, selector, out);
    return out;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

function matches(el, selector) {
  if (selector.startsWith("#")) return el.id === selector.slice(1);
  if (selector.startsWith(".")) return el.className.split(/\s+/).includes(selector.slice(1));
  if (selector.startsWith("[") && selector.endsWith("]")) {
    return el.attributes.has(selector.slice(1, -1));
  }
  throw new Error(`unsupported selector in DOM stub: ${selector}`);
}

function collect(root, selector, out) {
  for (const child of root.children) {
    if (matches(child, selector)) out.push(child);
    collect(child, selector, out);
  }
}

export function createDocument() {
  const document = new FakeElement("#document");
  document.createElement = (tag) => new FakeElement(tag);
  return document;
}

export { FakeElement };
