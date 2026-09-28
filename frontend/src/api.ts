export interface OceanEmbedRequest {
  latitude: number;
  longitude: number;
  date: string;
  depths: number[];
}

export interface DepthPrediction {
  depthM: number;
  temperatureC: number;
  uncertaintyC: number | null;
}

export interface SurfaceObservation {
  variable: string;
  value: number | null;
  unit: string;
}

export interface LiveStatus {
  latestUsableDate: string | null;
  inputWindowStart: string | null;
  inputWindowEnd: string | null;
  variablesReady: string[];
  variablesReadyCount: number;
  requiredVariablesCount: number;
  lastChecked: string | null;
  message: string | null;
  ready: boolean;
}

export interface SurfaceCoverage {
  ready: boolean;
  latitude: number;
  longitude: number;
  snappedLatitude: number | null;
  snappedLongitude: number | null;
  date: string;
  targetDate: string;
  windowStart: string;
  windowEnd: string;
  availableDates: string[];
  missingDates: string[];
  requiredVariables: number;
  variablesReady: number;
  readyVariables: string[];
  missingVariables: string[];
  message: string;
}

export interface OceanEmbedResponse {
  jobId: number;
  status: string;
  latitude: number;
  longitude: number;
  date: string;
  inputWindowStart?: string;
  inputWindowEnd?: string;
  modelVersion: string;
  gridResolutionDeg: number;
  createdAt: string;
  completedAt: string | null;
  errorMessage: string | null;
  predictions: DepthPrediction[];
  surfaceObservations?: SurfaceObservation[];

  depths: number[];
  temperature: number[];
  uncertainty: Array<number | null>;
  model_version: string;
}

type JsonRecord = Record<string, unknown>;

export type OceanEmbedErrorKind = "unsupported-location" | "request-failed";

export class OceanEmbedError extends Error {
  readonly kind: OceanEmbedErrorKind;

  constructor(kind: OceanEmbedErrorKind) {
    super(kind === "unsupported-location"
      ? "Location is not currently supported."
      : "The reconstruction could not be completed. Please try again.");
    this.name = "OceanEmbedError";
    this.kind = kind;
  }
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8080";

function containsUnsupportedGridError(value: unknown): boolean {
  if (typeof value === "string") {
    return value.toLowerCase().includes(
      "requested grid point could not be represented inside the 32x32 prediction region of the selected 64x64 tile",
    );
  }

  if (Array.isArray(value)) {
    return value.some(containsUnsupportedGridError);
  }

  if (value !== null && typeof value === "object") {
    return Object.values(value).some(containsUnsupportedGridError);
  }

  return false;
}

function parseResponseBody(responseText: string): unknown {
  try {
    return JSON.parse(responseText);
  } catch {
    return null;
  }
}

function isJsonRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function readField(record: JsonRecord, ...names: string[]): unknown {
  for (const name of names) {
    if (name in record) {
      return record[name];
    }
  }
  return undefined;
}

function requiredNumber(record: JsonRecord, ...names: string[]): number {
  const value = readField(record, ...names);
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OceanEmbedError("request-failed");
  }
  return value;
}

function requiredString(record: JsonRecord, ...names: string[]): string {
  const value = readField(record, ...names);
  if (typeof value !== "string") {
    throw new OceanEmbedError("request-failed");
  }
  return value;
}

function nullableNumber(record: JsonRecord, ...names: string[]): number | null {
  const value = readField(record, ...names);
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OceanEmbedError("request-failed");
  }
  return value;
}

function nullableString(record: JsonRecord, ...names: string[]): string | null {
  const value = readField(record, ...names);
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "string") {
    throw new OceanEmbedError("request-failed");
  }
  return value;
}

function readStringList(record: JsonRecord, ...names: string[]): string[] {
  const value = readField(record, ...names);
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

/**
 * Application dates must be real calendar days in exact YYYY-MM-DD form
 * with a non-zero-padded 4-digit year (rejects 0002-05-27, 0020-05-27, etc.).
 */
function isValidApplicationDate(value: string): boolean {
  if (!/^[1-9]\d{3}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }
  const [yearText, monthText, dayText] = value.split("-");
  const year = Number(yearText);
  const month = Number(monthText);
  const day = Number(dayText);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return (
    parsed.getUTCFullYear() === year
    && parsed.getUTCMonth() === month - 1
    && parsed.getUTCDate() === day
  );
}

function assertValidApplicationDate(value: string): void {
  if (!isValidApplicationDate(value)) {
    throw new Error(
      `Invalid application date: "${value}". Expected a real calendar date in YYYY-MM-DD format with a 4-digit year.`,
    );
  }
}

function shiftIsoDate(isoDate: string, days: number): string {
  assertValidApplicationDate(isoDate);
  const [year, month, day] = isoDate.split("-").map(Number);
  const shifted = new Date(Date.UTC(year, month - 1, day + days))
    .toISOString()
    .slice(0, 10);
  assertValidApplicationDate(shifted);
  return shifted;
}

function coverageRequestKey(
  latitude: number,
  longitude: number,
  date: string,
): string {
  return `${latitude}|${longitude}|${date}`;
}

const inflightCoverageRequests = new Map<string, Promise<SurfaceCoverage>>();

export async function fetchSurfaceCoverage(
  request: Pick<OceanEmbedRequest, "latitude" | "longitude" | "date">,
): Promise<SurfaceCoverage> {
  assertValidApplicationDate(request.date);

  const key = coverageRequestKey(
    request.latitude,
    request.longitude,
    request.date,
  );
  const existing = inflightCoverageRequests.get(key);
  if (existing) {
    return existing;
  }

  const promise = (async (): Promise<SurfaceCoverage> => {
    const query = new URLSearchParams({
      latitude: String(request.latitude),
      longitude: String(request.longitude),
      date: request.date,
    });
    const response = await fetch(
      `${API_BASE_URL}/api/v1/predictions/coverage?${query.toString()}`,
    );
    if (!response.ok) {
      throw new OceanEmbedError("request-failed");
    }

    let responseBody: unknown;
    try {
      responseBody = await response.json();
    } catch {
      throw new OceanEmbedError("request-failed");
    }
    if (!isJsonRecord(responseBody)) {
      throw new OceanEmbedError("request-failed");
    }

    const ready = readField(responseBody, "ready");
    if (typeof ready !== "boolean") {
      throw new OceanEmbedError("request-failed");
    }
    const date = nullableString(responseBody, "date") ?? request.date;
    return {
      ready,
      latitude: requiredNumber(responseBody, "latitude"),
      longitude: requiredNumber(responseBody, "longitude"),
      snappedLatitude: nullableNumber(responseBody, "snappedLatitude", "snapped_latitude"),
      snappedLongitude: nullableNumber(responseBody, "snappedLongitude", "snapped_longitude"),
      date,
      targetDate: nullableString(responseBody, "targetDate", "target_date") ?? date,
      windowStart: nullableString(responseBody, "windowStart", "window_start")
        ?? shiftIsoDate(date, -6),
      windowEnd: nullableString(responseBody, "windowEnd", "window_end") ?? date,
      availableDates: readStringList(responseBody, "availableDates", "available_dates"),
      missingDates: readStringList(responseBody, "missingDates", "missing_dates"),
      requiredVariables: requiredNumber(responseBody, "requiredVariables", "required_variables"),
      variablesReady: requiredNumber(responseBody, "variablesReady", "variables_ready"),
      readyVariables: readStringList(responseBody, "readyVariables", "ready_variables"),
      missingVariables: readStringList(responseBody, "missingVariables", "missing_variables"),
      message: requiredString(responseBody, "message"),
    };
  })().finally(() => {
    inflightCoverageRequests.delete(key);
  });

  inflightCoverageRequests.set(key, promise);
  return promise;
}

export async function fetchLiveStatus(): Promise<LiveStatus> {
  const response = await fetch(`${API_BASE_URL}/api/v1/live/status`);
  if (!response.ok) {
    throw new OceanEmbedError("request-failed");
  }

  let responseBody: unknown;
  try {
    responseBody = await response.json();
  } catch {
    throw new OceanEmbedError("request-failed");
  }

  if (!isJsonRecord(responseBody)) {
    throw new OceanEmbedError("request-failed");
  }

  const record = isJsonRecord(responseBody.data)
    ? responseBody.data
    : responseBody;
  const variablesReady = readStringList(
    record,
    "readyVariables",
    "ready_variables",
    "variablesReady",
    "variables_ready",
    "surfaceVariablesReady",
    "surface_variables_ready",
  );
  const rawVariablesReady = readField(record, "variablesReady", "variables_ready");
  const legacyCount = readField(
    record,
    "variablesReadyCount",
    "variables_ready_count",
    "surfaceVariablesReadyCount",
    "surface_variables_ready_count",
    "variableCount",
    "variable_count",
  );
  const variablesReadyCount = typeof rawVariablesReady === "number"
    && Number.isFinite(rawVariablesReady)
    ? rawVariablesReady
    : typeof rawVariablesReady === "boolean"
      ? rawVariablesReady ? 7 : 0
      : typeof legacyCount === "number" && Number.isFinite(legacyCount)
        ? legacyCount
        : variablesReady.length;
  const rawRequiredVariables = readField(
    record,
    "requiredVariables",
    "required_variables",
  );
  const legacyRequiredCount = readField(
    record,
    "requiredVariablesCount",
    "required_variables_count",
  );
  const requiredVariablesCount = typeof rawRequiredVariables === "number"
    && Number.isFinite(rawRequiredVariables)
    ? rawRequiredVariables
    : typeof legacyRequiredCount === "number" && Number.isFinite(legacyRequiredCount)
      ? legacyRequiredCount
      : 7;
  const rawReadiness = readField(record, "ready", "dataReady", "data_ready");
  const rawStatus = readField(record, "status");
  const readinessFlag = typeof rawReadiness === "boolean"
    ? rawReadiness
    : typeof rawVariablesReady === "boolean"
      ? rawVariablesReady
      : null;
  const latestUsableDate = nullableString(
    record,
    "latestUsableDate",
    "latest_usable_date",
    "latestDate",
    "latest_date",
  );
  const inputWindowStart = nullableString(
    record,
    "inputWindowStart",
    "input_window_start",
  );
  const inputWindowEnd = nullableString(
    record,
    "inputWindowEnd",
    "input_window_end",
  );
  const ready = readinessFlag
    ?? (typeof rawStatus === "string"
      ? rawStatus.toLowerCase() === "ready"
      : variablesReadyCount >= requiredVariablesCount);
  const message = nullableString(record, "message");
  const lastChecked = nullableString(record, "lastChecked", "last_checked");

  return {
    latestUsableDate,
    inputWindowStart,
    inputWindowEnd,
    variablesReady,
    variablesReadyCount,
    requiredVariablesCount,
    lastChecked,
    message,
    ready,
  };
}

function normalizePrediction(value: unknown): DepthPrediction {
  if (!isJsonRecord(value)) {
    throw new OceanEmbedError("request-failed");
  }
  return {
    depthM: requiredNumber(value, "depthM", "depth_m"),
    temperatureC: requiredNumber(value, "temperatureC", "temperature_c"),
    uncertaintyC: nullableNumber(
      value,
      "uncertaintyC",
      "uncertainty_c",
      "ensembleSpreadC",
      "ensemble_spread_c",
    ),
  };
}

function normalizeSurfaceObservation(value: unknown): SurfaceObservation {
  if (!isJsonRecord(value)) {
    throw new OceanEmbedError("request-failed");
  }
  return {
    variable: requiredString(value, "variable"),
    value: nullableNumber(value, "value"),
    unit: requiredString(value, "unit"),
  };
}

export async function reconstructOcean(
  request: OceanEmbedRequest
): Promise<OceanEmbedResponse> {
  assertValidApplicationDate(request.date);

  const response = await fetch(
    `${API_BASE_URL}/api/v1/predictions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    }
  );

  const responseText = await response.text();

  if (!response.ok) {
    const responseBody = parseResponseBody(responseText);
    const unsupportedLocation = containsUnsupportedGridError(responseBody)
      || containsUnsupportedGridError(responseText);
    throw new OceanEmbedError(
      unsupportedLocation ? "unsupported-location" : "request-failed",
    );
  }

  let parsedResponse: unknown;
  try {
    parsedResponse = JSON.parse(responseText);
  } catch {
    throw new OceanEmbedError("request-failed");
  }

  if (!isJsonRecord(parsedResponse)) {
    throw new OceanEmbedError("request-failed");
  }

  const backendResult = parsedResponse;
  const predictionsValue = readField(backendResult, "predictions");
  if (!Array.isArray(predictionsValue)) {
    throw new OceanEmbedError("request-failed");
  }
  const predictions = predictionsValue.map(normalizePrediction);

  const surfaceObservationsValue = readField(
    backendResult,
    "surfaceObservations",
    "surface_observations",
  );
  if (
    surfaceObservationsValue !== undefined
    && !Array.isArray(surfaceObservationsValue)
  ) {
    throw new OceanEmbedError("request-failed");
  }
  const surfaceObservations = Array.isArray(surfaceObservationsValue)
    ? surfaceObservationsValue.map(normalizeSurfaceObservation)
    : undefined;

  const inputWindowStart = nullableString(
    backendResult,
    "inputWindowStart",
    "input_window_start",
  );
  const inputWindowEnd = nullableString(
    backendResult,
    "inputWindowEnd",
    "input_window_end",
  );
  const modelVersion = requiredString(
    backendResult,
    "modelVersion",
    "model_version",
  );

  return {
    jobId: requiredNumber(backendResult, "jobId", "job_id"),
    status: requiredString(backendResult, "status"),
    latitude: requiredNumber(backendResult, "latitude"),
    longitude: requiredNumber(backendResult, "longitude"),
    date: requiredString(backendResult, "date"),
    inputWindowStart: inputWindowStart ?? undefined,
    inputWindowEnd: inputWindowEnd ?? undefined,
    modelVersion,
    gridResolutionDeg: requiredNumber(
      backendResult,
      "gridResolutionDeg",
      "grid_resolution_deg",
    ),
    createdAt: requiredString(backendResult, "createdAt", "created_at"),
    completedAt: nullableString(backendResult, "completedAt", "completed_at"),
    errorMessage: nullableString(backendResult, "errorMessage", "error_message"),
    surfaceObservations,
    predictions,

    depths: predictions.map(
      (prediction) => prediction.depthM
    ),

    temperature: predictions.map(
      (prediction) => prediction.temperatureC
    ),

    uncertainty: predictions.map(
      (prediction) => prediction.uncertaintyC
    ),

    model_version: modelVersion,
  };
}