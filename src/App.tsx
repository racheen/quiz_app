import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { useTheme } from './theme/useTheme';
import Home from './pages/Home';
import GlobalStyle from './theme/globalStyle';
import Footer from './components/Footer';

export default function App() {
  const { theme } = useTheme();

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
      </ThemeProvider>
    </>
  );
}
