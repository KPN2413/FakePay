/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // Temporary: don't block prod builds if ESLint config is missing
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.API_BASE_URL}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
