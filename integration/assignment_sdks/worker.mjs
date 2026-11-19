import { compileAllocation } from './node_modules/@increment/assignment/dist/index.js';
import { runCases } from './run_cases.mjs';

export default {
  async fetch(request) {
    return Response.json(await runCases(compileAllocation, await request.json()));
  },
};
