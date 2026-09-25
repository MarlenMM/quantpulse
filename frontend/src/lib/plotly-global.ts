/**
 * Plotly's source modules read Node's `global`; its prebuilt `dist` bundle
 * does not, which is why this was never needed until `lib/plotly.ts` started
 * importing the source to register only three trace types. Without it the
 * chart chunk throws "global is not defined" and no figure mounts. Imported
 * first by `lib/plotly.ts`, so it runs before any Plotly module does.
 */
const scope = globalThis as typeof globalThis & { global?: typeof globalThis };
scope.global ??= globalThis;

export {};
