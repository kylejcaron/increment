export async function runCases(compile, vectors) {
  const cases = [];
  for (const fixture of vectors.cases) {
    try {
      const evaluator = compile(vectors.configs[fixture.config]);
      const result = fixture.op === 'decide'
        ? await evaluator.decide(fixture.target, fixture.unit_id)
        : fixture.op === 'membership'
          ? await evaluator.holdoutMembership(fixture.target, fixture.unit_id)
          : null;
      cases.push({ id: fixture.id, result });
    } catch (error) {
      if (typeof error?.code !== 'string') throw error;
      cases.push({ id: fixture.id, error: error.code });
    }
  }
  return cases;
}
