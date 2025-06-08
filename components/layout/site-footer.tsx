import Link from "next/link";
import { Shield } from "lucide-react";

export function SiteFooter() {
  return (
    <footer className="border-t py-2">
      <div className="flex w-full flex-col md:flex-row items-center justify-between gap-2 px-4">
        <div className="flex flex-col items-center gap-1 md:flex-row md:gap-2">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <span className="font-medium">FakePay</span>
          </Link>
          <p className="text-center text-sm text-muted-foreground md:text-left">
            © 2025 FakePay. All rights reserved.
          </p>
        </div>
        <div className="flex gap-4 text-sm">
          <Link
            href="/privacy"
            className="hover:text-primary text-muted-foreground"
          >
            Privacy
          </Link>
          <Link
            href="/terms"
            className="hover:text-primary text-muted-foreground"
          >
            Terms
          </Link>
          <Link
            href="/contact"
            className="hover:text-primary text-muted-foreground"
          >
            Contact
          </Link>
        </div>
      </div>
    </footer>
  );
}
