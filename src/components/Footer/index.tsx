import { Container, CopyrightText, FooterText, GreenText } from './styles';

export const Footer = () => {
  return (
    <Container>
      <FooterText>
        developed and designed by <GreenText>Rachem</GreenText>
      </FooterText>
      <CopyrightText>Copyright © 2021</CopyrightText>
    </Container>
  );
};

export default Footer;
