import Link from "next/link";
import { Shield } from "lucide-react";

export function SiteFooter() {
  return (
    <footer className="border-t py-3">
      <div className="container flex flex-col items-center justify-between gap-2 md:h-20 md:flex-row">
        <div className="flex flex-col items-center gap-2 px-4 md:flex-row md:gap-1 md:px-0">
          <Link href="/" className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <span className="font-medium">FakePay</span>
          </Link>
          <p className="text-center text-sm text-muted-foreground md:text-left">
            © 2025 FakePay. All rights reserved.
          </p>
        </div>
        <div className="flex gap-3 px-4 md:px-0">
          <Link
            href="/privacy"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            Privacy
          </Link>
          <Link
            href="/terms"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            Terms
          </Link>
          <Link
            href="/contact"
            className="text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
          >
            Contact
          </Link>
        </div>
      </div>
    </footer>
  );
}
