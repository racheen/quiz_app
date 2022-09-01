import styled, { css } from 'styled-components';

type SectionProps = {
  full?: boolean;
  rightOnly?: boolean;
  both?: boolean;
};

export const Section = styled.div<SectionProps>`
  color: ${(props) => props.theme.colors.darkGray};
  padding-top: 20px;
  padding-bottom: 20px;
  min-height: 300px;

  ${(props) =>
    props.full &&
    css`
      padding-top: 0;
      min-height: 100vh;
      justify-content: center;
      align-items: center;
    `}

  ${(props) =>
    props.both &&
    css`
      background: linear-gradient(${props.theme.colors.lightGreen} 0 0)
        calc(0.25 * 100% / 8);
      background-size: 5px 90%;
      background-repeat: no-repeat;

      @media screen and ${props.theme.device.laptopL} {
        background: linear-gradient(${props.theme.colors.lightGreen} 0 0)
            calc(0.25 * 100% / 8),
          linear-gradient(${props.theme.colors.lightGreen} 0 0)
            calc(7.5 * 100% / 8);
        background-size: 10px 90%;
        background-repeat: no-repeat;
      } ;
    `}

  ${(props) =>
    props.rightOnly &&
    css`
      @media screen and ${props.theme.device.laptop} {
        background: linear-gradient(${props.theme.colors.lightGreen} 0 0)
          calc(7.5 * 100% / 8);
        background-size: 10px 90%;
        background-repeat: no-repeat;
      } ;
    `};

  @media screen and ${(props) => props.theme.device.tablet} {
    padding-top: 50px;
    padding-bottom: 50px;
  } ;
`;

export const SectionContentLayout = styled.div`
  display: flex;
  padding: 5px 50px 0 50px;
  flex-direction: column;

  @media screen and ${(props) => props.theme.device.tablet} {
    padding: 50px 100px 0 100px;
    flex-direction: row;
  } ;
`;

export const SectionContent = styled(SectionContentLayout)`
  flex-wrap: wrap;
  justify-content: space-evenly;
  align-items: center;

  p {
    font-size: ${(props) => props.theme.fontSizes.normal};
  }

  @media screen and ${(props) => props.theme.device.tablet} {
    align-items: flex-start;
    p {
      font-size: ${(props) => props.theme.fontSizes.large};
    }
  } ;
`;

export const TextDefault = styled(SectionContentLayout)`
  align-items: baseline;
  padding-top: 20px;
`;

export const Intro1 = styled(TextDefault)`
  padding: 250px 50px 0 50px;
  width: 200px;
  color: ${(props) => props.theme.colors.mediumGray};
  font-weight: ${(props) => props.theme.fontWeights.bold};
  flex-direction: row;

  @media screen and ${(props) => props.theme.device.tablet} {
    padding: 250px 100px 0 100px;
    width: 586px;
  } ;
`;

export const Intro2 = styled(Intro1)`
  display: block;
  padding: 0 0 0 50px;

  @media screen and ${(props) => props.theme.device.tablet} {
    padding: 0 0 0 100px;
  } ;
`;

export const H1 = styled.div`
  font-size: ${(props) => props.theme.fontSizes.h3};
  color: ${(props) => props.theme.colors.lightGreen};

  @media screen and (min-width: ${(props) => props.theme.size.tablet}) {
    font-size: ${(props) => props.theme.fontSizes.h1};
  } ;
`;

export const H4 = styled.div`
  font-size: ${(props) => props.theme.fontSizes.regular};

  @media screen and ${(props) => props.theme.device.tablet} {
    font-size: ${(props) => props.theme.fontSizes.h4};
  } ;
`;

export const H2 = styled.div`
  font-size: ${(props) => props.theme.fontSizes.h4};
  color: ${(props) => props.theme.colors.mediumGray};

  @media screen and ${(props) => props.theme.device.tablet} {
    font-size: ${(props) => props.theme.fontSizes.h2};
  } ;
`;

export const SectionTitle = styled(TextDefault)`
  font-size: ${(props) => props.theme.fontSizes.medium};
  color: ${(props) => props.theme.colors.lightGreen};
  font-weight: ${(props) => props.theme.fontWeights.bold};

  @media screen and ${(props) => props.theme.device.tablet} {
    font-size: ${(props) => props.theme.fontSizes.h4};
  } ;
`;

export const SectionText = styled(TextDefault)`
  font-weight: ${(props) => props.theme.fontWeights.mediumLight};
  font-size: ${(props) => props.theme.fontSizes.normal};
  color: ${(props) => props.theme.colors.darkGray};
  line-height: ${(props) => props.theme.fontSizes.h3};

  @media screen and ${(props) => props.theme.device.mobileL} {
    font-size: ${(props) => props.theme.fontSizes.medium};
    width: 700px;
  } ;
`;
