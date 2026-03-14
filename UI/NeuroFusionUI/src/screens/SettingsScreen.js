import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  SafeAreaView,
  Switch,
  Animated,
  Alert,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const SettingsScreen = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(true);
  
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnims = useRef([
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
    new Animated.Value(30),
  ]).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      ...slideAnims.map((anim, index) =>
        Animated.spring(anim, {
          toValue: 0,
          delay: index * 100,
          friction: 8,
          tension: 40,
          useNativeDriver: true,
        })
      ),
    ]).start();
  }, []);

  const handleGoBack = () => router.back();

  const handleSignOut = () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: () => {
            // Navigate to login screen
            navigation.navigate('Login');
          },
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />

      {/* Header */}
      <Animated.View style={[styles.header, { opacity: fadeAnim }]}>
        <View style={styles.statusBar}>
          <Text style={styles.time}>10:00</Text>
          <View style={styles.statusIcons}>
            <MaterialCommunityIcons name="signal" size={16} color="#000" />
            <MaterialCommunityIcons name="wifi" size={16} color="#000" style={{ marginLeft: 4 }} />
            <MaterialCommunityIcons name="battery" size={16} color="#000" style={{ marginLeft: 4 }} />
          </View>
        </View>

        <TouchableOpacity style={styles.backButton} onPress={handleGoBack}>
          <MaterialCommunityIcons name="arrow-left" size={24} color="#000" />
        </TouchableOpacity>

        <Text style={styles.headerTitle}>Settings</Text>
      </Animated.View>

      {/* Settings Options */}
      <View style={styles.settingsContainer}>
        {/* Dark Mode */}
        <Animated.View
          style={[
            styles.settingItem,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnims[0] }],
            },
          ]}
        >
          <Text style={styles.settingLabel}>Dark Mode</Text>
          <Switch
            value={darkMode}
            onValueChange={setDarkMode}
            trackColor={{ false: '#E0E0E0', true: '#B844FF' }}
            thumbColor={darkMode ? '#FFFFFF' : '#F4F3F4'}
            ios_backgroundColor="#E0E0E0"
          />
        </Animated.View>

        {/* Notifications */}
        <Animated.View
          style={[
            styles.settingItem,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnims[1] }],
            },
          ]}
        >
          <Text style={styles.settingLabel}>Notifications</Text>
          <Switch
            value={notifications}
            onValueChange={setNotifications}
            trackColor={{ false: '#E0E0E0', true: '#B844FF' }}
            thumbColor={notifications ? '#FFFFFF' : '#F4F3F4'}
            ios_backgroundColor="#E0E0E0"
          />
        </Animated.View>

        {/* About App */}
        <Animated.View
          style={[
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnims[2] }],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.settingItemNav}
            onPress={() => navigation.navigate('AboutApp')}
          >
            <Text style={styles.settingLabel}>About App</Text>
            <MaterialCommunityIcons name="chevron-right" size={24} color="#666" />
          </TouchableOpacity>
        </Animated.View>

        {/* User Information */}
        <Animated.View
          style={[
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnims[3] }],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.settingItemNav}
            onPress={() => navigation.navigate('UserInformation')}
          >
            <Text style={styles.settingLabel}>User Information</Text>
            <MaterialCommunityIcons name="chevron-right" size={24} color="#666" />
          </TouchableOpacity>
        </Animated.View>

        {/* Sign Out */}
        <Animated.View
          style={[
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnims[3] }],
            },
          ]}
        >
          <TouchableOpacity
            style={styles.settingItemNav}
            onPress={handleSignOut}
          >
            <Text style={styles.settingLabel}>Sign Out</Text>
            <MaterialCommunityIcons name="chevron-right" size={24} color="#666" />
          </TouchableOpacity>
        </Animated.View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  header: {
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  time: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000',
  },
  statusIcons: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F5F5F5',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#B844FF',
  },
  settingsContainer: {
    padding: 20,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  settingItemNav: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
});

export default SettingsScreen;