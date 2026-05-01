/**
 * Compile built-in extensions (TypeScript → CommonJS JS) and copy manifests.
 * Output: out/builtin-extensions/{id}/processor.js + manifest.json
 *
 * Two sources are pulled in:
 *   1. src/areas/workflows/nodes/   — TS-based workflow utility nodes
 *   2. extensions/modly-*-extension/ — vendored Python model adapters
 *
 * Both end up under the same out/builtin-extensions/{manifest.id}/ tree
 * and are synced to userData by syncBuiltinExtensions() at first launch.
 * Heavy assets (venvs, vendor/) are intentionally NOT included — those
 * get built on the user's machine via the extension's setup.py.
 */

import { execSync }                                           from 'child_process'
import { readdirSync, existsSync, readFileSync, cpSync, mkdirSync, statSync } from 'fs'
import { join, dirname }                                      from 'path'
import { fileURLToPath }                                      from 'url'

const root        = join(dirname(fileURLToPath(import.meta.url)), '..')
const srcDir      = join(root, 'src', 'areas', 'workflows', 'nodes')
const submodDir   = join(root, 'extensions')
const outDir      = join(root, 'out', 'builtin-extensions')

if (!existsSync(srcDir)) {
  console.log('[build-builtins] No builtin-extensions directory found, skipping.')
  process.exit(0)
}

// 1. Compile TypeScript
console.log('[build-builtins] Compiling TypeScript…')
execSync('npx tsc -p tsconfig.builtins.json', { cwd: root, stdio: 'inherit' })

// 2. Copy manifest.json, and optionally package.json + npm install
for (const id of readdirSync(srcDir)) {
  const extSrcDir = join(srcDir, id)
  if (!statSync(extSrcDir).isDirectory()) continue
  // Only process extension folders (those with a manifest.json)
  if (!existsSync(join(extSrcDir, 'manifest.json'))) continue

  const extOutDir = join(outDir, id)
  mkdirSync(extOutDir, { recursive: true })

  const manifestSrc = join(extSrcDir, 'manifest.json')
  if (existsSync(manifestSrc)) {
    cpSync(manifestSrc, join(extOutDir, 'manifest.json'))
    console.log(`[build-builtins] ${id}: manifest.json copied`)
  } else {
    console.warn(`[build-builtins] ${id}: manifest.json missing — skipping`)
  }

  const pkgSrc = join(extSrcDir, 'package.json')
  if (existsSync(pkgSrc)) {
    cpSync(pkgSrc, join(extOutDir, 'package.json'))
    console.log(`[build-builtins] ${id}: Installing npm dependencies…`)
    execSync('npm install --omit=dev --no-audit --no-fund', {
      cwd:   extOutDir,
      stdio: 'inherit',
    })
    console.log(`[build-builtins] ${id}: npm install done`)
  }

  // Copy any Python processor files
  for (const file of readdirSync(extSrcDir)) {
    if (file.endsWith('.py')) {
      cpSync(join(extSrcDir, file), join(extOutDir, file))
      console.log(`[build-builtins] ${id}: ${file} copied`)
    }
  }
}

// 3. Vendored model extensions (extensions/modly-*-extension/) → bundle as built-in
//    Destination subdir uses the manifest's `id`, NOT the submodule directory
//    name, because the runtime registers extensions by manifest.id.
if (existsSync(submodDir)) {
  for (const submoduleName of readdirSync(submodDir)) {
    const extSrc = join(submodDir, submoduleName)
    if (!statSync(extSrc).isDirectory()) continue
    const manifestPath = join(extSrc, 'manifest.json')
    if (!existsSync(manifestPath)) continue

    let manifest
    try {
      manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
    } catch (err) {
      console.warn(`[build-builtins] ${submoduleName}: malformed manifest — ${err.message}`)
      continue
    }
    const id = manifest.id
    if (!id) {
      console.warn(`[build-builtins] ${submoduleName}: manifest has no "id" — skipping`)
      continue
    }

    const extOutDir = join(outDir, id)
    mkdirSync(extOutDir, { recursive: true })

    // Code + metadata files. We deliberately skip vendor/, venv/, __pycache__/
    // — those are user-built and would balloon the .app size.
    const filesToCopy = ['manifest.json', 'generator.py', 'setup.py', 'build_vendor.py', 'README.md']
    for (const f of filesToCopy) {
      const srcPath = join(extSrc, f)
      if (existsSync(srcPath)) {
        cpSync(srcPath, join(extOutDir, f))
      }
    }
    console.log(`[build-builtins] ${submoduleName} → ${id} (model extension)`)
  }
}

console.log('[build-builtins] Done.')
