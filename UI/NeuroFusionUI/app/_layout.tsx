import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import {
  useFonts,
  RobotoSlab_300Light,
  RobotoSlab_600SemiBold,
} from '@expo-google-fonts/roboto-slab';


export default function RootLayout() {
  const [fontsLoaded] = useFonts({
    RobotoSlab_300Light,
    RobotoSlab_600SemiBold,
  });

  if (!fontsLoaded) {
    return null; // Prevent rendering until fonts load
  }
  
  return (
    <>
      <StatusBar style="auto" />
      <Stack
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen name="splash" />
        <Stack.Screen name="login" />
        <Stack.Screen name="signup" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="analyzing" />
        <Stack.Screen name="no-seizure-detected" />
        <Stack.Screen name="seizure-detected" />
        <Stack.Screen name="about-app" />
        <Stack.Screen name="user-information" />
      </Stack>
    </>
  );
}
