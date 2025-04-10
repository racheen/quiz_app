import { useEffect, useState } from "react";
import { defaultTheme, darkTheme } from "./themes";

type ThemeType = typeof defaultTheme;

const THEME_STORAGE_KEY = "default";

export const useTheme = () => {
  const [theme, setTheme] = useState<ThemeType>(darkTheme);

  // Load theme from localStorage on first load
  useEffect(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (savedTheme === "default") {
      setTheme(defaultTheme);
    } else {
      setTheme(darkTheme); // default fallback
    }
  }, []);

  const changeTheme = () => {
    const isCurrentlyDark = theme.id === darkTheme.id;

    const newTheme = isCurrentlyDark ? defaultTheme : darkTheme;
    setTheme(newTheme);
    localStorage.setItem(THEME_STORAGE_KEY, isCurrentlyDark ? "default" : "dark");
  };

  return { theme, changeTheme };
};
