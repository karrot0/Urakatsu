// What this build of the UI offers. Urakatsu's own app has every feature; a build that serves the UI
// from another service (one that runs it in the browser, say) replaces this module with its own and
// turns off what that service does not have.

export const features = {
  models: true,
  apiDocs: true,
  sources: true,
  knowledge: true,
  komga: true,
  automations: true,
  agent: true,
  workers: true,
  runtimes: true,
  translationProviders: true,
  hardware: true,
  updates: true,
  optionalStages: true,
};
