/// <reference types="vite/client" />

// Vite's ambient types, which every Vite + TypeScript scaffold carries and this
// project was missing. They declare the non-JS modules a bundler resolves but
// the type system cannot see on its own -- `declare module '*.css' {}` and
// friends -- plus `import.meta.env`.
//
// `vite/client` is not an `@types/*` package, so TypeScript does not pick it up
// automatically; it has to be referenced explicitly from a file inside
// `include` (this one) or listed in `compilerOptions.types`.
//
// Absent it, `import "./styles.css"` in `main.tsx` was an untyped side-effect
// import. TypeScript 5.9 let that pass silently; TypeScript 7 reports
// `TS2882: Cannot find module or type declarations for side-effect import`,
// which is what blocked the TypeScript 7 upgrade. The gap was always here --
// the newer compiler only stopped ignoring it.
