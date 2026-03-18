import React, { useEffect, useState } from 'react';
import { View, ActivityIndicator } from 'react-native';
import { Tabs, Redirect } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { onAuthStateChanged, User } from 'firebase/auth';
import { doc, getDoc } from 'firebase/firestore';
import { auth, db } from '../../src/config/firebase';
 
type Role = 'patient' | 'caretaker' | 'doctor';
 
const SCREEN_OPTIONS = {
  headerShown: false,
  tabBarActiveTintColor:   '#B844FF',
  tabBarInactiveTintColor: '#AAAAAA',
  tabBarStyle: {
    backgroundColor: '#FFFFFF',
    borderTopWidth:  1,
    borderTopColor:  '#EEEEEE',
    paddingTop:      6,
    paddingBottom:   8,
    height:          62,
  },
  tabBarLabelStyle: {
    fontSize:   11,
    fontWeight: '600' as const,
    marginTop:  2,
  },
};
 
const TabIcon = (iconName: string) =>
  ({ color, size }: { color: string; size: number }) => (
    <MaterialCommunityIcons name={iconName as any} size={size} color={color} />
  );
 
const LoadingScreen = () => (
  <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FFF' }}>
    <ActivityIndicator size="large" color="#B844FF" />
  </View>
);
 
function PatientTabs() {
  return (
    <Tabs screenOptions={SCREEN_OPTIONS}>
      <Tabs.Screen name="dashboard"          options={{ title: 'Home',     tabBarIcon: TabIcon('home')       }} />
      <Tabs.Screen name="analyzeEEG"         options={{ title: 'Analyze',  tabBarIcon: TabIcon('file-chart') }} />
      <Tabs.Screen name="history"            options={{ title: 'History',  tabBarIcon: TabIcon('history')    }} />
      <Tabs.Screen name="settings"           options={{ title: 'Settings', tabBarIcon: TabIcon('cog')        }} />
      <Tabs.Screen name="doctorDashboard"    options={{ href: null }} />
      <Tabs.Screen name="caretakerDashboard" options={{ href: null }} />
      <Tabs.Screen name="reports"            options={{href: null}} />
    </Tabs>
  );
}
 
function CaretakerTabs() {
  return (
    <Tabs screenOptions={SCREEN_OPTIONS}>
      <Tabs.Screen name="caretakerDashboard" options={{ title: 'Home',     tabBarIcon: TabIcon('home')       }} />
      <Tabs.Screen name="analyzeEEG"         options={{ title: 'Analyze',  tabBarIcon: TabIcon('file-chart') }} />
      <Tabs.Screen name="history"            options={{ title: 'History',  tabBarIcon: TabIcon('history')    }} />
      <Tabs.Screen name="settings"           options={{ title: 'Settings', tabBarIcon: TabIcon('cog')        }} />
      <Tabs.Screen name="dashboard"          options={{ href: null }} />
      <Tabs.Screen name="doctorDashboard"    options={{ href: null }} />
      <Tabs.Screen name="reports"            options={{href: null}} />
    </Tabs>
  );
}
 
function DoctorTabs() {
  return (
    <Tabs screenOptions={SCREEN_OPTIONS}>
      <Tabs.Screen name="doctorDashboard"    options={{ title: 'Home',     tabBarIcon: TabIcon('home')                  }} />
      <Tabs.Screen name="reports"            options={{ title: 'Reports',  tabBarIcon: TabIcon('chart-box')             }} />
      <Tabs.Screen name="history"            options={{ title: 'History',  tabBarIcon: TabIcon('history')               }} />
      <Tabs.Screen name="settings"           options={{ title: 'Settings', tabBarIcon: TabIcon('cog')                   }} />
      <Tabs.Screen name="dashboard"          options={{ href: null }} />
      <Tabs.Screen name="analyzeEEG"         options={{ href: null }} />
      <Tabs.Screen name="caretakerDashboard" options={{ href: null }} />
    </Tabs>
  );
}
 
export default function TabLayout() {
  const [user,    setUser]    = useState<User | null | undefined>(undefined);
  const [role,    setRole]    = useState<Role>('patient');
 
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        setUser(null);
        return;
      }
      setUser(currentUser);
      try {
        const snap = await getDoc(doc(db, 'users', currentUser.uid));
        if (snap.exists()) {
          setRole((snap.data()?.role as Role) ?? 'patient');
        }
      } catch {
        setRole('patient');
      }
    });
    return unsubscribe;
  }, []);
 
  // undefined = still loading, null = logged out, User = logged in
  if (user === undefined) return <LoadingScreen />;
  if (user === null)      return <Redirect href="/login" />;
 
  if (role === 'doctor')    return <DoctorTabs />;
  if (role === 'caretaker') return <CaretakerTabs />;
  return <PatientTabs />;
}