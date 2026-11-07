// The transport the gRPC client speaks over: gRPC-web to the Python service on the UI's own origin.
// Kept in its own module so a build that hosts the UI elsewhere can swap it for another transport
// (Connect's router transport serves the same service from inside the page).

import { createGrpcWebTransport } from "@connectrpc/connect-web";

export const transport = createGrpcWebTransport({
  baseUrl: window.location.origin,
  useBinaryFormat: true,
});
