import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SAM3 Colony Intelligence",
  description: "SAM3 기반 미생물 콜로니 검출 및 분석 워크벤치",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}