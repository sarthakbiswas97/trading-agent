"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/analytics", label: "Analytics" },
  { href: "/model", label: "Model" },
] as const;

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur-md border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 md:px-6 flex items-center justify-between h-14">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 via-cyan-500 to-violet-500 flex items-center justify-center">
            <span className="text-xs font-bold text-white leading-none">V</span>
          </div>
          <div>
            <span className="text-sm font-bold text-white tracking-tight">VAPM</span>
            <span className="hidden sm:inline text-xs text-gray-500 ml-1.5">Verifiable AI Portfolio Manager</span>
          </div>
        </Link>

        {/* Navigation Links */}
        <div className="flex items-center gap-1">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-gray-800 text-white"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-900"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
