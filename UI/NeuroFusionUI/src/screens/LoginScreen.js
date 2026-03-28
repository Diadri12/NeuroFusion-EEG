import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const ROLES = [
  { id: 'patient',   label: 'Patient',   icon: 'account'       },
  { id: 'caretaker', label: 'Caretaker', icon: 'account-heart' },
  { id: 'doctor',    label: 'Doctor',    icon: 'stethoscope'   },
];

const LoginScreen = ({ onLogin, onNavigateToSignUp }) => {
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [role,     setRole]     = useState('patient'); 

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      {/* <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" /> */}
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.content}>
          <View style={styles.logoContainer}>
            <MaterialCommunityIcons name="brain" size={60} color="#B844FF" />
            <Text style={styles.welcomeText}>Welcome to EpiGuard</Text>
            <Text style={styles.subtitleText}>Sign in to continue</Text>
          </View>

          <View style={styles.formContainer}>
            <Text style={styles.label}>I am a</Text>
            <View style={styles.roleRow}>
              {ROLES.map(r => (
                <TouchableOpacity
                  key={r.id}
                  onPress={() => setRole(r.id)}
                  style={[styles.roleBtn, role === r.id && styles.roleBtnActive]}
                  activeOpacity={0.75}
                >
                  <MaterialCommunityIcons
                    name={r.icon}
                    size={20}
                    color={role === r.id ? '#B844FF' : '#9CA3AF'}
                  />
                  <Text style={[styles.roleLabel, role === r.id && styles.roleLabelActive]}>
                    {r.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="example@xyz.com"
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />

            <Text style={styles.label}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="············"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
          </View>

          {/* ← pass role as third argument */}
          <TouchableOpacity style={styles.loginButton} onPress={() => onLogin(email, password, role)}>
            <Text style={styles.loginButtonText}>Login</Text>
          </TouchableOpacity>

          <View style={styles.signupContainer}>
            <Text style={styles.signupText}>Don't have an account? </Text>
            <TouchableOpacity onPress={onNavigateToSignUp}>
              <Text style={styles.signupLink}>Sign Up</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container:     { flex: 1, backgroundColor: '#FFFFFF' },
  scrollContent: { flexGrow: 1 },
  content:       { flex: 1, padding: 24, justifyContent: 'center' },
  logoContainer: { alignItems: 'center', marginBottom: 40 },
  welcomeText:   { fontSize: 24, fontWeight: 'bold', color: '#B844FF', marginTop: 16 },
  subtitleText:  { fontSize: 16, color: '#666666', marginTop: 4 },
  formContainer: { marginBottom: 24 },
  label:         { fontSize: 16, color: '#333333', marginBottom: 8, fontWeight: '500' },
  input: {
    borderWidth: 1, borderColor: '#E0E0E0', borderRadius: 12,
    padding: 16, fontSize: 16, marginBottom: 16, backgroundColor: '#F9F9F9',  color: '#333333',
  },
  loginButton:     { backgroundColor: '#B844FF', borderRadius: 28, padding: 16, alignItems: 'center', marginTop: 24 },
  loginButtonText: { color: '#FFFFFF', fontSize: 18, fontWeight: '600' },
  signupContainer: { flexDirection: 'row', justifyContent: 'center', marginTop: 20 },
  signupText:      { color: '#666666', fontSize: 14 },
  signupLink:      { color: '#B844FF', fontSize: 14, fontWeight: '600' },

  roleRow: { flexDirection: 'row', gap: 10, marginBottom: 20 },
  roleBtn: {
    flex: 1, paddingVertical: 10, borderRadius: 12,
    borderWidth: 1, borderColor: '#E0E0E0',
    backgroundColor: '#F9F9F9',
    alignItems: 'center', gap: 5,
  },
  roleBtnActive:   { borderColor: '#B844FF', backgroundColor: '#F5F0FF' },
  roleLabel:       { fontSize: 12, fontWeight: '500', color: '#9CA3AF' },
  roleLabelActive: { color: '#B844FF', fontWeight: '600' },
});

export default LoginScreen;