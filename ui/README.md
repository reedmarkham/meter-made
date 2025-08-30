# Next.js Parking Ticket Prediction App

This Next.js app predicts expired meter parking tickets in Washington, DC using machine learning and Google Maps API:

https://meter-made-ui-397485407730.us-east4.run.app/
![meter-made UI](screenshot.png)

## Deployment

The app is deployed using **Cloud Build** with a trigger connected to this repository. Changes pushed to the repository automatically trigger a build and deployment.

For more details, check the [Cloud Build documentation](https://cloud.google.com/build).

# More on Next.js

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
