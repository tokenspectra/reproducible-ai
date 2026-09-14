import { defineRailway, postgres, preserve, project, service } from "railway/iac";

/**
 * Infrastructure for the overversion demo backend.
 *
 * The API is built from a Dockerfile whose context is the repo root, because
 * services/api depends on packages/core by path; a service-rooted build could
 * not see it. Secrets (the SGLang key, the OpenRouter key, the token signing
 * seed) are set outside this file and marked preserve() so applying config
 * never wipes them.
 */
export default defineRailway(() => {
  const db = postgres("Postgres");

  const api = service("api", {
    build: { builder: "DOCKERFILE", dockerfilePath: "services/api/Dockerfile" },
    healthcheck: "/health",
    healthcheckTimeout: 60,
    domains: [{ domain: "api.overversion.com", port: 8000 }],
    env: {
      DATABASE_URL: db.env.DATABASE_URL,
      APP_ORIGIN: "https://app.overversion.com",
      INFER_BASE_URL: "https://infer.overversion.com",
      INFER_MODEL: "qwen3-8b",
      STATUS_INTERVAL_S: "10",
      INFER_API_KEY: preserve(),
      OPENROUTER_API_KEY: preserve(),
      TOKEN_SIGNING_SEED_HEX: preserve(),
    },
  });

  return project("overversion", { resources: [db, api] });
});
