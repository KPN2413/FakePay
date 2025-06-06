import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from '@/components/ui/sonner';
import { SiteHeader } from '@/components/layout/site-header';
import { SiteFooter } from '@/components/layout/site-footer';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'FakePay - UPI Scam Detection',
  description: 'Detect and prevent UPI payment scams with FakePay',
  keywords: ['UPI', 'scam detection', 'payment security', 'QR code verification'],
  authors: [{ name: 'FakePay Security Team' }],
  openGraph: {
    type: 'website',
    locale: 'en_IE',
    url: 'https://fakepay.vercel.app',
    title: 'FakePay - UPI Scam Detection',
    description: 'Detect and prevent UPI payment scams with FakePay',
    siteName: 'FakePay',
  },
}; 

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen bg-background antialiased`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="relative flex min-h-screen flex-col">
            <SiteHeader />
            <main className="flex-1">{children}</main>
            <SiteFooter />
          </div>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
