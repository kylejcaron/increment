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
    this.tagName = tagName.toLowerCase();
    this.attributes = new Map();
    this.children = [];
    this.listeners = {};
    this.parentElement = null;
    this._text = "";
    this._html = "";
    this._value = "";
    this.style = {};
    this.hidden = false;
    if (id !== undefined) this.setAttribute("id", id);
    if (className !== undefined) this.setAttribute("class", className);
    this.dataset = new Proxy(
      {},
      {
        get: (_target, key) => this.attributes.get(`data-${String(key)}`),
        set: (_target, key, value) => {
          this.setAttribute(`data-${String(key)}`, value);
          return true;
        },
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

  dispatch(type, event = {}) {
    const dispatched = { ...event, target: event.target ?? this };
    for (const handler of this.listeners[type] ?? []) handler(dispatched);
  }

  append(...nodes) {
    for (const node of nodes) {
      if (node.parentElement) {
        node.parentElement.children = node.parentElement.children.filter((child) => child !== node);
      }
      node.parentElement = this;
      this.children.push(node);
    }
  }

  replaceChildren(...nodes) {
    for (const child of this.children) child.parentElement = null;
    this.children = [];
    this.append(...nodes);
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
      toggle: (name, force) => {
        const present = names().includes(name);
        const add = force ?? !present;
        if (add) self.classList.add(name);
        else self.classList.remove(name);
        return add;
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
    this.replaceChildren();
    this._text = String(value);
    this._html = "";
  }

  get innerHTML() {
    return this._html;
  }

  set innerHTML(value) {
    this.replaceChildren();
    this._html = String(value);
    this._text = "";
  }

  querySelectorAll(selector) {
    const out = [];
    collect(this, selector, out);
    return out;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  closest(selector) {
    for (let node = this; node; node = node.parentElement) {
      if (matches(node, selector)) return node;
    }
    return null;
  }

  contains(node) {
    for (let child = node; child; child = child.parentElement) {
      if (child === this) return true;
    }
    return false;
  }

  remove() {
    if (this.parentElement) {
      this.parentElement.children = this.parentElement.children.filter((child) => child !== this);
      this.parentElement = null;
    }
  }

  focus() {}

  getBoundingClientRect() {
    return { width: 0, height: 0, top: 0, right: 0, bottom: 0, left: 0 };
  }
}

function matches(el, selector) {
  return selector.split(",").some((part) => {
    const trimmed = part.trim();
    const compound = trimmed.match(/^([a-zA-Z][\w-]*)?\[([^\]=]+)(?:=["']?([^"'\]]+)["']?)?\]$/);
    if (compound) {
      const [, tag, attribute, value] = compound;
      return (
        (!tag || el.tagName === tag.toLowerCase()) &&
        el.attributes.has(attribute) &&
        (value === undefined || el.getAttribute(attribute) === value)
      );
    }
    if (trimmed.startsWith("#")) return el.id === trimmed.slice(1);
    if (trimmed.startsWith(".")) return el.className.split(/\s+/).includes(trimmed.slice(1));
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
      const [attribute, value] = trimmed.slice(1, -1).split("=");
      return (
        el.attributes.has(attribute) &&
        (value === undefined || el.getAttribute(attribute) === value.replace(/^["']|["']$/g, ""))
      );
    }
    return el.tagName === trimmed.toLowerCase();
  });
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
  document.createTextNode = (value) => {
    const node = new FakeElement("#text");
    node.nodeType = 3;
    node.nodeValue = String(value);
    node.textContent = String(value);
    return node;
  };
  document.querySelectorAll = (selector) => FakeElement.prototype.querySelectorAll.call(document, selector);
  document.querySelector = (selector) => document.querySelectorAll(selector)[0] ?? null;
  document.getElementById = (id) => document.querySelector(`#${id}`);
  return document;
}

export { FakeElement };
