import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { useTheme } from './theme/useTheme';
import Home from './pages/Home';
import GlobalStyle from './theme/globalStyle';
import Footer from './components/Footer';
import { ThemeToggle } from './components/ThemeToggle';
import styled from 'styled-components';

const MainWrapper = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh; /* Ensures the wrapper takes at least full viewport height */
  justify-content: space-between; /* Ensures footer is pushed to the bottom */
`;

export default function App() {
  const { theme, changeTheme } = useTheme();
  const isDark = theme.id === 1;

  return (
    <>
      <ThemeProvider theme={theme}>
        <GlobalStyle />
        <MainWrapper>
          <Router>
            <Routes>
              <Route path='/' element={<Home />} />
            </Routes>
          </Router>

          <ThemeToggle isDark={isDark} onToggle={changeTheme} />
          <Footer />
        </MainWrapper>
      </ThemeProvider>
    </>
  );
}
