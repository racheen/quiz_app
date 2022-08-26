import { useEffect, useState } from "react";
import { defaultTheme, darkTheme } from "./themes";

export const useTheme = () => {
  const [theme, setTheme] = useState(darkTheme);

  function changeTheme() {
    const themeId = theme.id;
    switch (themeId) {
      case 1:
        return setTheme(defaultTheme);
      case 2:
        return setTheme(darkTheme);
    }
  }

  useEffect(() => {}, [theme]);

  return { theme, changeTheme };
};
