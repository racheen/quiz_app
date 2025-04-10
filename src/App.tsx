import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { useTheme } from './theme/useTheme';
import Home from './pages/Home';
import GlobalStyle from './theme/globalStyle';
import Footer from './components/Footer';
import { ThemeToggle } from './components/ThemeToggle';

export default function App() {
  const { theme, changeTheme } = useTheme();
  const isDark = theme.id === 2;

  return (
    <>
      <ThemeProvider theme={theme}>
        <GlobalStyle />
        <Router>
          <Routes>
            <Route path='/' element={<Home />} />
          </Routes>
        </Router>
        <Footer />
        <ThemeToggle isDark={isDark} onToggle={changeTheme} />
      </ThemeProvider>
    </>
  );
}
