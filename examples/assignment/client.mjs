async function decideLocally() {
  const { compileAllocation } = await import('@increment/assignment');
  const response = await fetch('/config.json');
  if (!response.ok) throw new Error('Could not load allocation configuration');
  const evaluator = compileAllocation(await response.json());
  return evaluator.decide('checkout', document.body.dataset.demoUnit);
}

const fromServer = JSON.parse(document.querySelector('#server-decision').textContent);
const decision = fromServer ?? await decideLocally();
const output = document.querySelector('output');
output.textContent = `${decision.status}: ${decision.arm ?? 'baseline'}`;
output.dataset.status = decision.status;
// Record real exposure here if a variant is actually shown. Deciding alone logs nothing.
