import { Stack, useRouter } from 'expo-router';
import { useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
const API_URL = "http://192.168.0.111:5000";

export default function Layout() {
  const router = useRouter();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = await AsyncStorage.getItem('token');
        if (!token) {
          router.replace('/signup');
        }
      } catch (error) {
        console.error('Error checking authentication:', error);
        router.replace('/signup');
      }
    };
    checkAuth();
  }, []);

  return (
    <Stack
      screenOptions={{
        headerShown: false,
        animation: 'none'
      }}
    />
  );
}
