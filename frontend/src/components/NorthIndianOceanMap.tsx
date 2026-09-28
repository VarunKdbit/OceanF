import type { MouseEvent } from "react";

interface NorthIndianOceanMapProps {
  latitude: number | null;
  longitude: number | null;
  onSelect: (latitude: number, longitude: number) => void;
}

const DOMAIN = {
  latitudeMin: 5,
  latitudeMax: 30,
  longitudeMin: 45,
  longitudeMax: 105,
} as const;

const VIEW = {
  left: 78,
  top: 72,
  width: 654,
  height: 280,
} as const;

const LANDMASSES = [
  {
    name: "Arabia",
    points: [
      [45, 30], [61, 30], [60, 28], [59, 26], [58, 24], [57, 23],
      [56, 22], [55, 20], [54, 19], [53, 18], [51, 17], [50, 16],
      [48, 15], [46, 15.5], [45, 16.5],
    ],
  },
  {
    name: "Indian subcontinent",
    points: [
      [68, 30], [74, 30], [75, 28], [74, 26], [72.5, 24.5], [71.5, 23],
      [70, 22], [68.5, 23], [67.5, 24], [67.5, 22], [68.5, 20],
      [69.5, 19], [70, 17.5], [71, 16], [72, 14], [72.8, 12],
      [74, 10], [75, 8.5], [77, 7.5], [78.5, 8.5], [80, 10],
      [80.8, 12], [81.2, 14], [82.5, 16], [83.5, 18], [85, 19.5],
      [87, 21], [89, 22], [91, 22], [92, 24], [94, 26], [96, 28],
      [99, 30],
    ],
  },
  {
    name: "Myanmar",
    points: [
      [99.543309, 20.186598],
      [98.959676, 19.752981],
      [98.253724, 19.708203],
      [97.797783, 18.62708],
      [97.375896, 18.445438],
      [97.859123, 17.567946],
      [98.493761, 16.837836],
      [98.903348, 16.177824],
      [98.537376, 15.308497],
      [98.192074, 15.123703],
      [98.430819, 14.622028],
      [99.097755, 13.827503],
      [99.212012, 13.269294],
      [99.196354, 12.804748],
      [99.587286, 11.892763],
      [99.038121, 10.960546],
      [98.553551, 9.93296],
      [98.457174, 10.675266],
      [98.764546, 11.441292],
      [98.428339, 12.032987],
      [98.509574, 13.122378],
      [98.103604, 13.64046],
      [97.777732, 14.837286],
      [97.597072, 16.100568],
      [97.16454, 16.928734],
      [96.505769, 16.427241],
      [95.369352, 15.71439],
      [94.808405, 15.803454],
      [94.188804, 16.037936],
      [94.533486, 17.27724],
      [94.324817, 18.213514],
      [93.540988, 19.366493],
      [93.663255, 19.726962],
      [93.078278, 19.855145],
      [92.368554, 20.670883],
      [92.303234, 21.475485],
      [92.652257, 21.324048],
      [92.672721, 22.041239],
      [93.166128, 22.27846],
      [93.060294, 22.703111],
      [93.286327, 23.043658],
      [93.325188, 24.078556],
      [94.106742, 23.850741],
      [94.552658, 24.675238],
      [94.603249, 25.162495],
      [95.155153, 26.001307],
      [95.124768, 26.573572],
      [96.419366, 27.264589],
      [97.133999, 27.083774],
      [97.051989, 27.699059],
      [97.402561, 27.882536],
      [97.327114, 28.261583],
      [97.911988, 28.335945],
      [98.246231, 27.747221],
      [98.68269, 27.508812],
      [98.712094, 26.743536],
      [98.671838, 25.918703],
      [97.724609, 25.083637],
      [97.60472, 23.897405],
      [98.660262, 24.063286],
      [98.898749, 23.142722],
      [99.531992, 22.949039],
      [99.240899, 22.118314],
      [99.983489, 21.742937],
      [100.416538, 21.558839],
      [101.150033, 21.849984],
      [101.180005, 21.436573],
      [100.329101, 20.786122],
      [100.115988, 20.41785],
      [99.543309, 20.186598]
    ],
  },

] as const;

const ISLANDS = [
  [[79.5, 9.8], [81.8, 9.8], [81.4, 6.2], [80.5, 5.8]],
  [[73.2, 13.5], [73.5, 13.5], [73.4, 12.8], [73.1, 12.9]],
  [[72.2, 10.5], [72.4, 10.5], [72.3, 9.8], [72.1, 9.9]],
] as const;

function project(latitude: number, longitude: number) {
  return {
    x: VIEW.left
      + ((longitude - DOMAIN.longitudeMin)
        / (DOMAIN.longitudeMax - DOMAIN.longitudeMin)) * VIEW.width,
    y: VIEW.top
      + ((DOMAIN.latitudeMax - latitude)
        / (DOMAIN.latitudeMax - DOMAIN.latitudeMin)) * VIEW.height,
  };
}

function polygonPoints(points: ReadonlyArray<readonly [number, number]>) {
  return points
    .map(([longitude, latitude]) => {
      const point = project(latitude, longitude);
      return `${point.x},${point.y}`;
    })
    .join(" ");
}

function coordinateLabel(value: number, axis: "latitude" | "longitude") {
  const degrees = Number(value.toFixed(2));
  return axis === "latitude"
    ? `${Math.abs(degrees)}Â°${degrees >= 0 ? "N" : "S"}`
    : `${Math.abs(degrees)}Â°${degrees >= 0 ? "E" : "W"}`;
}

export default function NorthIndianOceanMap({
  latitude,
  longitude,
  onSelect,
}: NorthIndianOceanMapProps) {
  const hasSelectedPoint =
    latitude !== null
    && longitude !== null
    && latitude >= DOMAIN.latitudeMin
    && latitude <= DOMAIN.latitudeMax
    && longitude >= DOMAIN.longitudeMin
    && longitude <= DOMAIN.longitudeMax;
  const selected = hasSelectedPoint
    ? project(latitude, longitude)
    : null;
  const selectedLabel = latitude !== null && longitude !== null
    ? `${coordinateLabel(latitude, "latitude")} Â· ${coordinateLabel(longitude, "longitude")}`
    : "";
  const longitudeFineGrid = Array.from({ length: 241 }, (_, index) => {
    const value = DOMAIN.longitudeMin + index * 0.25;
    return {
      value,
      x: project(DOMAIN.latitudeMin, value).x,
    };
  });
  const latitudeFineGrid = Array.from({ length: 101 }, (_, index) => {
    const value = DOMAIN.latitudeMin + index * 0.25;
    return {
      value,
      y: project(value, DOMAIN.longitudeMin).y,
    };
  });
  const longitudeGuides = Array.from({ length: 13 }, (_, index) => {
    const value = DOMAIN.longitudeMin + index * 5;
    return {
      value,
      x: project(DOMAIN.latitudeMin, value).x,
    };
  });
  const latitudeGuides = Array.from({ length: 6 }, (_, index) => {
    const value = DOMAIN.latitudeMin + index * 5;
    return {
      value,
      y: project(value, DOMAIN.longitudeMin).y,
    };
  });

  function selectFromMap(event: MouseEvent<SVGSVGElement>) {
    const svg = event.currentTarget;
    const matrix = svg.getScreenCTM();
    if (!matrix) return;

    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const localPoint = point.matrixTransform(matrix.inverse());
    const longitudeValue = DOMAIN.longitudeMin
      + ((localPoint.x - VIEW.left) / VIEW.width)
        * (DOMAIN.longitudeMax - DOMAIN.longitudeMin);
    const latitudeValue = DOMAIN.latitudeMax
      - ((localPoint.y - VIEW.top) / VIEW.height)
        * (DOMAIN.latitudeMax - DOMAIN.latitudeMin);

    if (
      latitudeValue >= DOMAIN.latitudeMin
      && latitudeValue <= DOMAIN.latitudeMax
      && longitudeValue >= DOMAIN.longitudeMin
      && longitudeValue <= DOMAIN.longitudeMax
    ) {
      onSelect(
        Number(latitudeValue.toFixed(2)),
        Number(longitudeValue.toFixed(2)),
      );
    }
  }

  return (
    <svg
      className="ocean-map"
      viewBox="0 0 800 420"
      role="img"
      aria-label="Scientific schematic of the OceanEmbed North Indian Ocean domain, 5 to 30 degrees north and 45 to 105 degrees east, on a 0.25 degree grid. Click within the domain to select a coordinate."
      onClick={selectFromMap}
    >
      <defs>
        <linearGradient id="oceanMapBackground" x1="0" y1="0" x2="0.9" y2="1">
          <stop offset="0%" stopColor="#06325a" />
          <stop offset="54%" stopColor="#0b4f83" />
          <stop offset="100%" stopColor="#052443" />
        </linearGradient>
        <linearGradient id="oceanMapLand" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#8bd63f" />
          <stop offset="100%" stopColor="#55b936" />
        </linearGradient>
        <radialGradient id="oceanMapProbeGlow">
          <stop offset="0%" stopColor="#a7f7ff" stopOpacity="0.42" />
          <stop offset="48%" stopColor="#52ddf2" stopOpacity="0.2" />
          <stop offset="100%" stopColor="#52ddf2" stopOpacity="0" />
        </radialGradient>
        <clipPath id="oceanMapDomainClip">
          <rect
            x={VIEW.left}
            y={VIEW.top}
            width={VIEW.width}
            height={VIEW.height}
          />
        </clipPath>
      </defs>

      <rect width="800" height="420" fill="url(#oceanMapBackground)" />
      <path
        d="M0 55 C110 26 194 58 276 40 S442 38 522 55 S696 24 800 49"
        className="map-current-contour"
      />
      <path
        d="M0 391 C138 366 237 402 369 382 S609 371 800 395"
        className="map-current-contour secondary"
      />

      <text className="map-title" x="28" y="29">
        LIVE RECONSTRUCTION REGION
      </text>
      <text className="map-region-name" x="28" y="48">
        NORTH INDIAN OCEAN
      </text>
      <text className="map-domain-meta" x="772" y="37" textAnchor="end">
        5â€“30Â°N  |  45â€“105Â°E  |  0.25Â° GRID
      </text>

      <g clipPath="url(#oceanMapDomainClip)">
        {longitudeFineGrid.map((line) => (
          <line
            key={`fine-lon-${line.value}`}
            x1={line.x}
            x2={line.x}
            y1={VIEW.top}
            y2={VIEW.top + VIEW.height}
            className="map-grid-line fine"
          />
        ))}
        {latitudeFineGrid.map((line) => (
          <line
            key={`fine-lat-${line.value}`}
            x1={VIEW.left}
            x2={VIEW.left + VIEW.width}
            y1={line.y}
            y2={line.y}
            className="map-grid-line fine"
          />
        ))}

        {longitudeGuides.map((line) => (
          <line
            key={`guide-lon-${line.value}`}
            x1={line.x}
            x2={line.x}
            y1={VIEW.top}
            y2={VIEW.top + VIEW.height}
            className="map-grid-line guide"
          />
        ))}
        {latitudeGuides.map((line) => (
          <line
            key={`guide-lat-${line.value}`}
            x1={VIEW.left}
            x2={VIEW.left + VIEW.width}
            y1={line.y}
            y2={line.y}
            className="map-grid-line guide"
          />
        ))}

        {LANDMASSES.map((land) => (
          <polygon
            key={land.name}
            points={polygonPoints(land.points)}
            className="map-land"
            fill="url(#oceanMapLand)"
          />
        ))}
        {ISLANDS.map((island, index) => (
          <polygon
            key={`island-${index}`}
            points={polygonPoints(island)}
            className="map-land island"
            fill="url(#oceanMapLand)"
          />
        ))}

        <text className="map-water-label arabian-sea" {...project(16, 61)}>
          ARABIAN SEA
        </text>
        <text className="map-water-label bay-of-bengal" {...project(16, 88)}>
          BAY OF BENGAL
        </text>
        <text className="map-water-label indian-ocean" {...project(8, 77)}>
          INDIAN OCEAN
        </text>
        <text className="map-land-label arabia-label" {...project(25, 52)}>
          ARABIA
        </text>
        <text className="map-land-label india-label" {...project(23, 96)}>
          MYANMAR
        </text>

        {selected && (
          <g className="map-probe" aria-hidden="true">
            <circle
              className="map-selected-glow"
              cx={selected.x}
              cy={selected.y}
              r="23"
            />
            <line
              className="map-selected-crosshair"
              x1={selected.x - 15}
              x2={selected.x - 6}
              y1={selected.y}
              y2={selected.y}
            />
            <line
              className="map-selected-crosshair"
              x1={selected.x + 6}
              x2={selected.x + 15}
              y1={selected.y}
              y2={selected.y}
            />
            <line
              className="map-selected-crosshair"
              x1={selected.x}
              x2={selected.x}
              y1={selected.y - 15}
              y2={selected.y - 6}
            />
            <line
              className="map-selected-crosshair"
              x1={selected.x}
              x2={selected.x}
              y1={selected.y + 6}
              y2={selected.y + 15}
            />
            <circle
              className="map-selected-halo"
              cx={selected.x}
              cy={selected.y}
              r="7"
            />
            <circle
              className="map-selected-point"
              cx={selected.x}
              cy={selected.y}
              r="3.2"
            />
            <text
              className="map-selected-label"
              x={selected.x > VIEW.left + VIEW.width - 130
                ? selected.x - 12
                : selected.x + 12}
              y={selected.y < VIEW.top + 36
                ? selected.y + 24
                : selected.y - 12}
              textAnchor={selected.x > VIEW.left + VIEW.width - 130
                ? "end"
                : "start"}
            >
              {selectedLabel}
            </text>
          </g>
        )}
      </g>

      <rect
        className="map-domain-border"
        x={VIEW.left}
        y={VIEW.top}
        width={VIEW.width}
        height={VIEW.height}
      />
      {longitudeGuides.map((line) => (
        <text
          key={`lon-label-${line.value}`}
          className="map-axis-label"
          x={line.x}
          y={VIEW.top + VIEW.height + 18}
          textAnchor="middle"
        >
          {coordinateLabel(line.value, "longitude")}
        </text>
      ))}
      {latitudeGuides.map((line) => (
        <text
          key={`lat-label-${line.value}`}
          className="map-axis-label"
          x={VIEW.left - 10}
          y={line.y + 3}
          textAnchor="end"
        >
          {coordinateLabel(line.value, "latitude")}
        </text>
      ))}
    </svg>
  );
}